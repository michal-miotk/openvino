// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ====================================================================
// Zfuzowany blok:  conv1(KxK) -> +bias1 -> SiLU -> conv2(1x1) -> +bias2 -> SiLU
// ====================================================================
//
// Kernel jest wariacja na temat convolution_gpu_bfyx_f16.cl - uzywa tego
// samego mapowania "jeden lane sub-group = jeden kanal w 16-kanalowym
// slice fsv16", tych samych odczytow blokowych i tego samego line_cache.
// Roznica polega na tym, ze CALY blok dwoch konwolucji liczony jest w
// jednym uruchomieniu kernela, bez materializowania tensora posredniego
// w pamieci globalnej.
//
// Dlaczego to w ogole da sie zfuzowac
// -----------------------------------
// Druga konwolucja jest 1x1 ze stride 1, wiec piksel wyjsciowy (x, y)
// zalezy WYLACZNIE od piksela posredniego (x, y) - zero halo, zero
// przeliczania. Ale zalezy od WSZYSTKICH kanalow posrednich, a jedna
// sub-group liczy tylko 16 z nich. Dlatego:
//
//   * work-group sklada sie z NUM_SUB_GROUPS sub-group i pokrywa
//     WSZYSTKIE slice'y kanalow (posrednich i wyjsciowych) dla jednego
//     kafelka przestrzennego (OUTPUT_X_BLOCK_SIZE pikseli X, jeden Y,
//     jeden batch),
//   * faza 1: sub-group `sg` liczy slice'y posrednie sg, sg+NUM_SUB_GROUPS,
//     ... i po biasie + SiLU zapisuje je do SLM,
//   * barrier,
//   * faza 2: sub-group `sg` liczy slice'y wyjsciowe sg, sg+NUM_SUB_GROUPS,
//     ... czytajac tensor posredni juz tylko z SLM.
//
// Tensor posredni nigdy nie dotyka pamieci globalnej - to jest caly zysk
// tej fuzji (oszczedzone jest jedno pelne zapisanie i jedno pelne
// odczytanie tensora o rozmiarze B*MID_F*Y*X, plus dwa launche kernela).
//
// Ograniczenia (pilnowane po stronie hosta, patrz
// fused_conv_silu_pair_kernel_bfyx_f16.cpp):
//   * conv2 musi byc 1x1, stride 1, bez paddingu i dilation,
//   * brak konwolucji grupowanej,
//   * wejscie i wyjscie w b_fs_yx_fsv16, wszystkie tensory tego samego typu,
//   * MID_IC_BLOCKS * OUTPUT_X_BLOCK_SIZE * 16 elementow musi zmiescic
//     sie w SLM.

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"

// Aliasy typu elementu i jego wariantow wektorowych - dokladnie jak w
// convolution_gpu_bfyx_f16.cl. Wagi i biasy sa tu zwyklymi wejsciami
// (INPUT1..INPUT4), a nie tensorami `weights`/`bias` kernel_selectora,
// wiec wszedzie uzywamy rodziny DT_INPUT_* zamiast DT_FILTER_*/DT_BIAS_*.
// Host wymusza, zeby wszystkie piec wejsc mialo ten sam typ danych.
#define INPUT_TYPE        INPUT0_TYPE
#define INPUT_TYPE2       MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8       MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)

// GET_SRC(data, id) rozglasza wartosc trzymana przez lane `id` do
// wszystkich lane'ow sub-group. Dla typow 16-bitowych shuffle nie dziala
// wprost na typie danych, wiec wartosc jedzie przez ushort.
#if INPUT0_TYPE_SIZE == 2
#   define AS_INPUT_SRC         CAT(as_, MAKE_VECTOR_TYPE(INPUT_TYPE, OUTPUT_X_BLOCK_SIZE))
#   define AS_US_SRC            CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
#   define GET_SRC(data, id)    AS_INPUT_SRC(_sub_group_shuffle(AS_US_SRC(data), id))
#else
#   define GET_SRC(data, id)    _sub_group_shuffle(data, id)
#endif

// SiLU / Swish: x * sigmoid(beta * x). Dziala elementowo na wektorach.
// Beta jest stala JIT-owa (dla klasycznego SiLU rowna 1.0).
#define SILU1(v)  ((v) / (INPUT0_VAL_ONE + exp(-TO_INPUT0_TYPE(SWISH_BETA_1) * (v))))
#define SILU2(v)  ((v) / (INPUT0_VAL_ONE + exp(-TO_INPUT0_TYPE(SWISH_BETA_2) * (v))))

// Liczba kanalow w jednym slice fsv16 = rozmiar sub-group.
#define FEATURE_SLICE_SIZE 16

// Rozmiar bufora SLM na tensor posredni: dla kazdego slice'a kanalow
// posrednich trzymamy OUTPUT_X_BLOCK_SIZE pikseli po 16 kanalow.
#define MID_SLM_SIZE (MID_IC_BLOCKS * OUTPUT_X_BLOCK_SIZE * FEATURE_SLICE_SIZE)

// Indeks w buforze SLM. Uklad to [slice][kanal][piksel], czyli piksele
// kafelka leza ciagle. Dzieki temu:
//   * faza 1: kazdy lane zapisuje swoje OUTPUT_X_BLOCK_SIZE wynikow pod
//     kolejne adresy (kompilator sklada to w jeden zapis wektorowy),
//   * faza 2: adres nie zalezy od lane'a, wiec caly wektor pikseli dla
//     danego kanalu posredniego jest jednym broadcastowym odczytem
//     wektorowym, zamiast OUTPUT_X_BLOCK_SIZE odczytow skalarnych.
#define MID_SLM_IDX(fs, c, i) ((((fs) * FEATURE_SLICE_SIZE) + (c)) * OUTPUT_X_BLOCK_SIZE + (i))

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, WORK_GROUP_SIZE, 1)))
KERNEL(fused_conv_silu_pair_gpu_bfyx_f16)(
    __global INPUT0_TYPE* input,
    __global INPUT1_TYPE* weights1,
    __global INPUT2_TYPE* bias1,
    __global INPUT3_TYPE* weights2,
    __global INPUT4_TYPE* bias2,
    __global OUTPUT_TYPE* output)
{
    // Indeks lane'a w sub-group = kanal w obrebie biezacego slice'a fsv16.
    const int sglid = get_sub_group_local_id();
    // Numer sub-group w work-group. Faza 1 i faza 2 rozdzielaja miedzy nie
    // slice'y kanalow (odpowiednio posrednich i wyjsciowych).
    const int sg = (int)get_local_id(1) / SUB_GROUP_SIZE;

    const int b = (uint)get_global_id(2);

    // get_global_id(0) numeruje splaszczone pary (y, kafelek-x).
    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    // Wektor akumulatora: jedna wartosc na kazda pozycje X w kafelku.
    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE) vec_t;

    // Lewy gorny rog pola recepcyjnego conv1 dla tego kafelka.
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    const int input_spatial_size_x = INPUT0_SIZE_X;

    // Ile pozycji zbuforowanej linii wypada poza realnym wejsciem (czyli
    // musi byc potraktowane jako zero-padding).
    const int right_unreachable_count_x = min(max(0, input_x + INPUT_LINE_SIZE - input_spatial_size_x),
                                              INPUT_LINE_SIZE);
    const int left_unreachable_count_x = min(max(0, -input_x), INPUT_LINE_SIZE);

    // --- Pitch'e wejscia w layoucie b_fs_yx_fsv16 -----------------------
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    // Baza adresu bez skladowej Y - wiersz doliczamy dopiero w petli po kh,
    // z jawnym clampem (patrz nizej).
    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              INPUT0_PAD_BEFORE_SIZE_Y * input_y_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch;

    // --- Pitch'e wyjscia w layoucie b_fs_yx_fsv16 -----------------------
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_base = b * output_b_pitch +
                             output_fs_pad_before * output_fs_pitch +
                             (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                             (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;

    // --- Pitch'e wag ----------------------------------------------------
    // Oba zestawy wag sa przepakowane po stronie transformacji do layoutu
    // os_is_yx_isv16_osv16 i podane jako plaskie bufory 1-D, wiec kernel
    // adresuje je wprost tymi samymi wzorami co convolution_gpu_bfyx_f16.
    const uint filter1_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter1_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter1_y_pitch = filter1_x_pitch * FILTER1_SIZE_X;
    const uint filter1_is_pitch = filter1_y_pitch * FILTER1_SIZE_Y;
    const uint filter1_os_pitch = filter1_is_pitch * IC_BLOCKS;

    // conv2 jest 1x1, wiec wymiary przestrzenne filtra znikaja i krok po
    // slice'ach kanalow wejsciowych to po prostu jeden blok 16x16.
    const uint filter2_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter2_is_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter2_os_pitch = filter2_is_pitch * MID_IC_BLOCKS;

    // Tensor posredni (po conv1 + bias1 + SiLU) dla tego kafelka.
    __local INPUT_TYPE mid_slm[MID_SLM_SIZE];

    // ================================================================
    // FAZA 1: conv1 -> +bias1 -> SiLU  ->  SLM
    // ================================================================
    // Kazda sub-group bierze co NUM_SUB_GROUPS-ty slice kanalow
    // posrednich. Gdy MID_IC_BLOCKS <= NUM_SUB_GROUPS, petla wykonuje sie
    // najwyzej raz i czesc sub-group nie robi w tej fazie nic.
    for (int mfs = sg; mfs < MID_IC_BLOCKS; mfs += NUM_SUB_GROUPS) {
        // Akumulator startuje od biasu kanalu wyjsciowego conv1 tego lane'a,
        // rozgloszonego na wszystkie pozycje X kafelka.
        vec_t dst = (vec_t)(DT_INPUT_BLOCK_READ(bias1, mfs * FEATURE_SLICE_SIZE));

        const uint filter1_offset = mfs * filter1_os_pitch;

        // Redukcja po slice'ach kanalow WEJSCIOWYCH.
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
            __attribute__((opencl_unroll_hint(FILTER1_SIZE_Y)))
            for (int kh = 0; kh < FILTER1_SIZE_Y; kh++) {
                // Wiersze filtra w calosci poza wejsciem wnosza zero -
                // taniej je pominac niz buforowac i mnozyc zera.
                const int in_y = input_y + kh * DILATION_SIZE_Y;
                if (in_y < 0 || in_y >= INPUT0_SIZE_Y)
                    continue;

                // UWAGA: adres liczymy z jawnie ograniczonym in_y. Straznik
                // powyzej i tak odrzuca wiersze spoza obrazu, ale gdy
                // IC_BLOCKS == 1 kompilator (IGC) zwija ten fragment do
                // prostego bloku i potrafi zrobic if-conversion, czyli
                // wykonac odczyt spekulacyjnie mimo `continue`. Dla in_y < 0
                // adres przekreca sie na uint i laduje ~4 GB poza buforem, co
                // konczy sie CL_OUT_OF_RESOURCES. Clamp jest zwyklym
                // dzialaniem arytmetycznym, wiec obowiazuje takze na sciezce
                // spekulacyjnej i trzyma adres w granicach bufora.
                const uint in_y_safe = (uint)clamp(in_y, 0, INPUT0_SIZE_Y - 1);

                const uint input_row_base = mad24((uint)icb, input_fs_pitch,
                        input_offset + in_y_safe * input_y_pitch);

                // Jeden wiersz wejscia zbuforowany raz i reuzyty dla
                // wszystkich FILTER1_SIZE_X pozycji kw.
                INPUT_TYPE line_cache[INPUT_LINE_SIZE];

#if INPUT_LEFTOVERS
                // Ostatni, czesciowo wypelniony slice kanalow wejscia:
                // czytamy skalarnie, zeby dalo sie osobno wyzerowac kanaly
                // paddingowe (ich zawartosc w pamieci jest nieokreslona).
                if ((icb + 1) * FEATURE_SLICE_SIZE >= INPUT0_FEATURE_NUM) {
                    for (int xb = 0; xb < INPUT_LINE_SIZE; xb++) {
                        const int in_x = input_x + xb;
                        if (icb * FEATURE_SLICE_SIZE + sglid >= INPUT0_FEATURE_NUM ||
                            in_x < 0 || in_x >= input_spatial_size_x)
                            line_cache[xb] = INPUT0_VAL_ZERO;
                        else
                            line_cache[xb] = input[mad24((uint)xb, input_x_pitch, input_row_base) + sglid];
                    }
                }
                else
#endif  // INPUT_LEFTOVERS
                {
                    // Szybka sciezka: caly slice jest realny, wiec ida
                    // szerokie odczyty blokowe; jawnie zerujemy tylko
                    // marginesy wypadajace poza obrazem.
                    int xb = 0;
                    for (int i = 0; i < left_unreachable_count_x; i++)
                        line_cache[xb + i] = INPUT0_VAL_ZERO;
                    xb += left_unreachable_count_x;

                    const int reachable_size = INPUT_LINE_SIZE - right_unreachable_count_x;
                    for (; xb + 8 <= reachable_size; xb += 8) {
                        INPUT_TYPE8 vv = DT_INPUT_BLOCK_READ8(input, mad24((uint)xb, input_x_pitch, input_row_base));
                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                        line_cache[xb + 4] = vv[4];
                        line_cache[xb + 5] = vv[5];
                        line_cache[xb + 6] = vv[6];
                        line_cache[xb + 7] = vv[7];
                    }
                    for (; xb + 4 <= reachable_size; xb += 4) {
                        INPUT_TYPE4 vv = DT_INPUT_BLOCK_READ4(input, mad24((uint)xb, input_x_pitch, input_row_base));
                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                    }
                    for (; xb < reachable_size; xb++) {
                        line_cache[xb] = DT_INPUT_BLOCK_READ(input, mad24((uint)xb, input_x_pitch, input_row_base));
                    }
                    // Reszta linii (prawy margines poza obrazem, a przy bardzo
                    // malych wejsciach takze wszystko co zostalo) to zera.
                    for (; xb < INPUT_LINE_SIZE; xb++)
                        line_cache[xb] = INPUT0_VAL_ZERO;
                }

                __attribute__((opencl_unroll_hint(FILTER1_SIZE_X)))
                for (int kw = 0; kw < FILTER1_SIZE_X; kw++) {
                    // Wybieramy z linii OUTPUT_X_BLOCK_SIZE wartosci wejscia
                    // (po jednej na pozycje X kafelka), z uwzglednieniem
                    // stride i dilation.
                    vec_t src;
                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
                    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if FILTER1_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * i];
#endif
                    }

                    // Kazdy lane laduje 16 wag laczacych JEGO kanal wyjsciowy
                    // conv1 ze wszystkimi 16 kanalami wejscia slice'a icb -
                    // dwa odczyty blokowe po 8.
                    const uint filter1_base = mad24((uint)icb, filter1_is_pitch,
                            mad24((uint)kh, filter1_y_pitch,
                                    mad24((uint)kw, filter1_x_pitch, filter1_offset)));
                    INPUT_TYPE8 wei0 = DT_INPUT_BLOCK_READ8(weights1, filter1_base);
                    INPUT_TYPE8 wei1 = DT_INPUT_BLOCK_READ8(weights1,
                            mad24(8u, filter1_isv_pitch, filter1_base));

                    // Rozglaszamy kazdy z 16 kanalow wejscia do wszystkich
                    // lane'ow, zeby kazdy lane mogl domknac swoj iloczyn
                    // skalarny dla wlasnego kanalu wyjsciowego.
                    const vec_t src0  = GET_SRC(src, 0);
                    const vec_t src1  = GET_SRC(src, 1);
                    const vec_t src2  = GET_SRC(src, 2);
                    const vec_t src3  = GET_SRC(src, 3);
                    const vec_t src4  = GET_SRC(src, 4);
                    const vec_t src5  = GET_SRC(src, 5);
                    const vec_t src6  = GET_SRC(src, 6);
                    const vec_t src7  = GET_SRC(src, 7);
                    const vec_t src8  = GET_SRC(src, 8);
                    const vec_t src9  = GET_SRC(src, 9);
                    const vec_t src10 = GET_SRC(src, 10);
                    const vec_t src11 = GET_SRC(src, 11);
                    const vec_t src12 = GET_SRC(src, 12);
                    const vec_t src13 = GET_SRC(src, 13);
                    const vec_t src14 = GET_SRC(src, 14);
                    const vec_t src15 = GET_SRC(src, 15);

                    dst = mad(wei0.s0, src0,  dst);
                    dst = mad(wei0.s1, src1,  dst);
                    dst = mad(wei0.s2, src2,  dst);
                    dst = mad(wei0.s3, src3,  dst);
                    dst = mad(wei0.s4, src4,  dst);
                    dst = mad(wei0.s5, src5,  dst);
                    dst = mad(wei0.s6, src6,  dst);
                    dst = mad(wei0.s7, src7,  dst);
                    dst = mad(wei1.s0, src8,  dst);
                    dst = mad(wei1.s1, src9,  dst);
                    dst = mad(wei1.s2, src10, dst);
                    dst = mad(wei1.s3, src11, dst);
                    dst = mad(wei1.s4, src12, dst);
                    dst = mad(wei1.s5, src13, dst);
                    dst = mad(wei1.s6, src14, dst);
                    dst = mad(wei1.s7, src15, dst);
                }
            }
        }

        // bias1 jest juz w akumulatorze - zostaje aktywacja.
        dst = SILU1(dst);

#if MID_LEFTOVERS
        // Ogon ostatniego slice'a kanalow posrednich to kanaly-paddingowe.
        // Zerujemy je, zeby faza 2 nie wciagnela do sumy smieci (wagi conv2
        // dla tych kanalow tez sa wyzerowane przy pakowaniu, ale zerowanie
        // po tej stronie jest tansze i niezalezne od tamtego zalozenia).
        if (mfs * FEATURE_SLICE_SIZE + sglid >= MID_FEATURE_NUM)
            dst = INPUT0_VAL_ZERO;
#endif

        __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++)
            mid_slm[MID_SLM_IDX(mfs, sglid, i)] = dst[i];
    }

    // Tensor posredni musi byc kompletny, zanim ktorakolwiek sub-group
    // zacznie po nim redukowac w fazie 2.
    barrier(CLK_LOCAL_MEM_FENCE);

    // ================================================================
    // FAZA 2: conv2 (1x1) -> +bias2 -> SiLU  ->  wyjscie
    // ================================================================
    for (int ofs = sg; ofs < OC_BLOCKS; ofs += NUM_SUB_GROUPS) {
        vec_t dst2 = (vec_t)(DT_INPUT_BLOCK_READ(bias2, ofs * FEATURE_SLICE_SIZE));

        const uint filter2_offset = ofs * filter2_os_pitch;

        // Redukcja po WSZYSTKICH kanalach posrednich - to jest ten wymiar,
        // przez ktory fuzja w ogole potrzebuje SLM i bariery.
        for (int m = 0; m < MID_IC_BLOCKS; m++) {
            const uint filter2_base = mad24((uint)m, filter2_is_pitch, filter2_offset);
            INPUT_TYPE8 w2a = DT_INPUT_BLOCK_READ8(weights2, filter2_base);
            INPUT_TYPE8 w2b = DT_INPUT_BLOCK_READ8(weights2,
                    mad24(8u, filter2_isv_pitch, filter2_base));

            __attribute__((opencl_unroll_hint(FEATURE_SLICE_SIZE)))
            for (int c = 0; c < FEATURE_SLICE_SIZE; c++) {
                // Waga kanalu posredniego `c` dla kanalu wyjsciowego tego lane'a.
                // Maska & 7 trzyma indeks komponentu w zakresie obu polowek
                // niezaleznie od tego, ktora galaz zostanie wybrana.
                const INPUT_TYPE w = (c < 8) ? w2a[c & 7] : w2b[c & 7];

                // Wartosci posrednie kanalu `c` dla calego kafelka X. Adres
                // nie zalezy od lane'a, wiec to jest broadcast z SLM.
                const __local INPUT_TYPE* mid_row = mid_slm + MID_SLM_IDX(m, c, 0);
                vec_t sv;
                __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
                for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++)
                    sv[i] = mid_row[i];

                dst2 = mad(w, sv, dst2);
            }
        }

        dst2 = SILU2(dst2);

        typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) out_vec_t;
        const uint output_offset = output_base + (uint)ofs * output_fs_pitch;

#if OUTPUT_LEFTOVERS
        // Slice kanalow wyjscia zawiera kanaly-paddingowe - zapis skalarny
        // ze straznikiem na kanale i na pozycji X.
        if ((ofs + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
                if ((ofs * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X)
                    output[output_offset + i * output_x_pitch + sglid] = TO_OUTPUT_TYPE(dst2[i]);
            }
        }
        else
#endif  // OUTPUT_LEFTOVERS
        {
            out_vec_t res;
            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++)
                res[i] = TO_OUTPUT_TYPE(dst2[i]);

            if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0) {
                // Caly kafelek miesci sie w wyjsciu - jeden szeroki zapis blokowy.
#if OUTPUT_X_BLOCK_SIZE == 8
                DT_OUTPUT_BLOCK_WRITE8(output, output_offset, res);
#elif OUTPUT_X_BLOCK_SIZE == 4
                DT_OUTPUT_BLOCK_WRITE4(output, output_offset, res);
#elif OUTPUT_X_BLOCK_SIZE == 2
                DT_OUTPUT_BLOCK_WRITE2(output, output_offset, res);
#elif OUTPUT_X_BLOCK_SIZE == 1
                DT_OUTPUT_BLOCK_WRITE(output, output_offset, res);
#else
#   error fused_conv_silu_pair_gpu_bfyx_f16.cl: unsupported output x block size
#endif
            } else {
                // Ostatni kafelek w wierszu wystaje poza wyjscie - zapisujemy
                // tylko wazne poczatkowe pozycje.
                for (int i = 0; i < OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE; i++)
                    DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, res[i]);
            }
        }
    }
}

// Sprzatamy makra zdefiniowane na gorze, zeby nie wyciekly do kolejnych
// kerneli sklejanych w tej samej jednostce translacji.
#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8
#undef AS_INPUT_SRC
#undef AS_US_SRC
#undef GET_SRC
#undef SILU1
#undef SILU2
#undef FEATURE_SLICE_SIZE
#undef MID_SLM_SIZE
#undef MID_SLM_IDX
