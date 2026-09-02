// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Ten kernel implementuje konwolucje dla layoutow blokowych "fsv16"
// (b_fs_yx_fsv16 dla wejscia/wyjscia). W tych layoutach wymiar FEATURE
// (kanal) jest podzielony na "slice'y" po 16 kolejnych kanalow, a wewnatrz
// jednego slice'a 16 wartosci kanalow dla danego piksela lezy w pamieci
// obok siebie (contiguous). Dzieki temu jedna OpenCL-owa sub-group
// (rozmiar 16, patrz REQD_SUB_GROUP_SIZE nizej) mozna zmapowac bezposrednio
// na jeden slice kanalow: kazdy work-item (lane) w sub-group "posiada"
// dokladnie jeden kanal, co pozwala na szybkie, ciagle odczyty/zapisy
// blokowe oraz tanie broadcasty miedzy lane'ami przez sub_group_shuffle,
// zamiast rozproszonego dostepu do pamieci.
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
// LSC (Load/Store Cache) software prefetch intrinsics - same mechanism
// oneDNN's own GPU kernels use (see
// thirdparty/onednn_gpu/src/gpu/intel/include/tile_ops.h,
// cooperative_prefetch_2d / __builtin_IB_lsc_prefetch_global_*).
//
// UWAGA: nie da sie tego dostac przez
// #include "include/batch_headers/tile_ops.cl" - mechanizm "batch headers"
// w kernel_selectorze dolacza dany batch header tylko do tych kerneli,
// ktore juz go realnie uzywaly (rejestracja per-kernel), a nie do kazdego
// kernela ktory go #include'uje w zrodle .cl. Dla tego kernela deklaracje
// nigdy nie trafialy do faktycznie kompilowanego przez IGC zrodla, mimo ze
// tresc tile_ops.cl byla poprawnie obecna w wygenerowanym
// ks_primitive_db_batch_headers.inc - stad realny blad kompilacji
// clBuildProgram: "use of unknown builtin
// '__builtin_IB_lsc_prefetch_global_uint'" / "use of undeclared identifier
// 'LSC_LDCC_L1C_L3C'". Zamiast tego deklarujemy potrzebne elementy wprost
// tutaj (identyczne z tile_ops.cl), kernel jest wiec samowystarczalny.
//
// UWAGA 2: kernel_selector kompiluje wiele instancji JIT-owanych (rozne
// zestawy parametrow) TEGO SAMEGO kernela w jednym wspolnym "batchu" -
// wszystkie ich teksty zrodlowe trafiaja do JEDNEGO wywolania
// clBuildProgram jako jeden duzy plik. Bez header-guarda ta deklaracja
// pojawialaby sie w nim wielokrotnie -> "redefinition of 'LSC_LDCC'".
// Standardowy include-guard dziala tu poprawnie, bo caly sklejony batch
// i tak przechodzi przez jeden przebieg preprocesora.
#ifndef GATHER_PREFETCH_LSC_LDCC_DEFINED
#define GATHER_PREFETCH_LSC_LDCC_DEFINED
enum LSC_LDCC {
    LSC_LDCC_DEFAULT = 0,
    LSC_LDCC_L1UC_L3UC = 1,
    LSC_LDCC_L1UC_L3C = 2,
    LSC_LDCC_L1C_L3UC = 3,
    LSC_LDCC_L1C_L3C = 4,
    LSC_LDCC_L1S_L3UC = 5,
    LSC_LDCC_L1S_L3C = 6,
    LSC_LDCC_L1IAR_L3C = 7,
};

extern void __builtin_IB_lsc_prefetch_global_uint(
        const __global uint *base, int immElemOff, enum LSC_LDCC cacheOpt);
#endif // GATHER_PREFETCH_LSC_LDCC_DEFINED


// Aliasy dla typu elementu wejscia i jego wariantow wektorowych.
// OUTPUT_X_BLOCK_SIZE (definiowane przez generator kernela po stronie
// hosta, nie w tym pliku) mowi ile pikseli wyjsciowych wzdluz X liczy
// jeden work-item na raz - czyli jak "szeroki" jest wektor akumulatora.
#define INPUT_TYPE        INPUT0_TYPE
#define INPUT_TYPE2       MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8       MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)

// Wektor 8 elementow wagi (filtra) - odpowiada granulacji uzywanej przez
// DT_FILTER_BLOCK_READ8 nizej (kazdy odczyt blokowy pobiera 8 wag na lane).
#define FILTER_TYPE8      MAKE_VECTOR_TYPE(FILTER_TYPE, 8)

// Pomocnicze makra "as_<typ>" do reinterpretacji bitowej (np. potraktowanie
// ushort jako half bez konwersji wartosci) - potrzebne, bo niektore funkcje
// pomocnicze block-read/shuffle dzialaja tylko na typach calkowitych, wiec
// dane fp16 musza chwilowo "udawac" ushort, zeby przez nie przejsc, a potem
// wrocic do wlasciwego typu.
#define AS_INPUT_TYPE     CAT(as_, INPUT_TYPE)
#define AS_INPUT_TYPE2    CAT(as_, INPUT_TYPE2)
#define AS_INPUT_TYPE4    CAT(as_, INPUT_TYPE4)
#define AS_INPUT_TYPE8    CAT(as_, INPUT_TYPE8)

#define AS_FILTER_TYPE8   CAT(as_, FILTER_TYPE8)

// Pomocnicze makra specyficzne dla formatu wyjscia, potrzebne tylko gdy
// tensor wyjsciowy uzywa zwyklego (nieblokowanego) layoutu bfyx zamiast
// blokowego layoutu fsv.
#if OUTPUT_FORMAT_BFYX
    // Typ wektorowy wyjscia o rozmiarze rownym liczbie zapisywanych na raz
    // pikseli X.
#   define OUTPUTVTYPE(n)       CAT(OUTPUT_TYPE, n)
#   define TO_OUTPUTVTYPE       CAT(convert_, OUTPUTVTYPE(OUTPUT_X_BLOCK_SIZE))
    // vstoreN - zapisuje N kolejnych elementow wyjscia zaczynajac od wskaznika.
#   define VSTORE               CAT(vstore, OUTPUT_X_BLOCK_SIZE)
#endif  // OUTPUT_FORMAT_BFYX

// GET_SRC(data, id) rozglasza (broadcast) wartosc trzymana przez lane `id`
// biezacej sub-group do wszystkich lane'ow. To jest kluczowy element,
// dzieki ktoremu layout "jeden lane = jeden kanal wejscia" zamienia sie
// w przeplyw danych "kazdy lane potrzebuje wszystkich kanalow wejscia do
// policzenia swojego iloczynu skalarnego dla wlasnego kanalu wyjscia"
// (patrz duzy komentarz nizej, przy `GET_SRC(src, 0..15)`).
// Dla typow 16-bitowych (fp16) sub_group_shuffle nie dziala bezposrednio
// na tym typie danych, wiec wartosc jest rzutowana bitowo na ushort,
// przesylana przez shuffle, i rzutowana z powrotem.
#if INPUT0_TYPE_SIZE == 2
#   define AS_INPUT_SRC         CAT(as_, MAKE_VECTOR_TYPE(INPUT_TYPE, OUTPUT_X_BLOCK_SIZE))
#   define AS_US_SRC            CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
#   define GET_SRC(data, id)    AS_INPUT_SRC(_sub_group_shuffle(AS_US_SRC(data), id))
#else
#   define GET_SRC(data, id)    _sub_group_shuffle(data, id)
#endif

// Liczba kanalow w jednym "slice" cech w layoucie fsv16. Jest rowna
// rozmiarowi sub-group: jeden lane <-> jeden kanal w biezacym slice.
#define FEATURE_SLICE_SIZE 16

// Zaokraglenie liczby kanalow wyjscia/wejscia w gore do wielokrotnosci
// rozmiaru slice'a, bo layout blokowy zawsze alokuje cale slice'y (ogon
// ostatniego slice'a moze zawierac kanaly-paddingowe).
#define FILTER_OFM_NUM_ALIGNED (((FILTER_OFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)
#define FILTER_IFM_NUM_ALIGNED (((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE) * FEATURE_SLICE_SIZE)

// Rozmiar sub-group musi byc rowny FEATURE_SLICE_SIZE (16), zeby mapowanie
// lane <-> kanal opisane wyzej mialo sens.
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
// Ksztalt work-group: 1 x (SUB_GROUP_SIZE * SLM_DIV_FACTOR) x 1.
// SLM_DIV_FACTOR > 1 oznacza, ze kilka sub-group wspolpracuje (przez
// pamiec lokalna, patrz partial_summ nizej), zeby policzyc jeden slice
// kanalow wyjscia - kazda sub-group liczy tylko czesc kanalow wejscia,
// wiec czesciowe sumy trzeba na koncu polaczyc przez SLM.
__attribute__((reqd_work_group_size(1, SUB_GROUP_SIZE * SLM_DIV_FACTOR, 1)))
KERNEL(convolution_bfyx_f16)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    // Indeks lane'a w obrebie sub-group. W tym layoucie identyfikuje, za
    // ktory pojedynczy kanal (z biezacego 16-kanalowego slice'a) ten
    // work-item odpowiada - jako kanal wyjsciowy, ktory produkuje, albo
    // jako kanal wejsciowy, ktorego wartosc aktualnie trzyma i rozglasza.
    const int sglid = get_sub_group_local_id();
    // Indeks batcha - jedna grupa work-itemow na pare (kafelek X, slice
    // kanalow, batch).
    const int b = (uint)get_global_id(2);

    // get_global_id(0) numeruje pary (y, kafelek-x) splaszczone razem:
    // przechodzi przez wszystkie X_BLOCKS kafelkow jednego wiersza zanim
    // przejdzie do nastepnego wiersza, wiec x/y odtwarzamy przez
    // dzielenie/modulo wzgledem X_BLOCKS.
    const int xy = get_global_id(0);
    // Startowa wspolrzedna X wyjscia dla kafelka tego work-itemu (kazdy
    // work-item produkuje OUTPUT_X_BLOCK_SIZE kolejnych pikseli wyjscia
    // wzdluz X).
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);
    const int input_spatial_size_x = INPUT0_SIZE_X;

    // Lokalny id w wymiarze 1 (wymiar "feature" work-group), uzywany
    // nizej zeby ustalic, za ktora czesc (ulamek) kanalow wejscia
    // odpowiada ta sub-group, gdy SLM_DIV_FACTOR > 1.
    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    // Ktory 16-kanalowy slice kanalow wyjscia liczy cala ta work-group.
    const int feature_block = (int)get_group_id(1);

#if GROUPED
    // --- Rachunki dla konwolucji grupowej (grouped convolution) -------
    // W konwolucji grupowej kanaly wejscia/wyjscia sa podzielone na
    // niezalezne grupy, kazda z wlasnym zestawem wag. 16-kanalowy slice
    // wyjscia moze "rozjechac sie" na granicy grup, gdy FILTER_OFM_NUM
    // (liczba kanalow wyjscia na grupe) nie jest wielokrotnoscia 16.
    // Ponizszy blok ustala, dla slice'a 16 kanalow wyjscia tej work-group,
    // ktore grupy on obejmuje i ile ich jest.
    //
    // `group`: indeks pierwszej (najnizszej kanalowo) grupy dotykanej
    // przez ten slice.
    const int group = (feature_block * FEATURE_SLICE_SIZE) / FILTER_OFM_NUM;
    // Ile kanalow wyjscia grupy `group` zostaje "za" poczatkiem tego slice'a.
    const int prev_group_leftover = (FILTER_OFM_NUM * (group + 1)) - (feature_block * FEATURE_SLICE_SIZE);
    // Liczba kolejnych grup, ktorych kanaly miesza sie w tym 16-elementowym slice.
    int groups_per_sub_group = 1;
    if (prev_group_leftover < 16)
        groups_per_sub_group += ((FEATURE_SLICE_SIZE - prev_group_leftover - 1) / FILTER_OFM_NUM) + 1;
    // Do ktorej grupy nalezy kanal wyjsciowy TEGO konkretnego lane'a.
    const uint my_group = group + (sglid / FILTER_OFM_NUM);
#else
    // Konwolucja niegrupowa (albo jedna grupa): wszystko jest "grupa 0"
    // i slice nigdy nie przechodzi przez granice.
    const int group = 0;
    const int groups_per_sub_group = 1;
#endif  // GROUPED

    // Wektor akumulatora na lane: jedna czesciowo policzona wartosc
    // wyjscia na kazda pozycje X w kafelku wyjsciowym tego work-itemu.
    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, OUTPUT_X_BLOCK_SIZE) vec_t;

    // Lewy gorny rog (we wspolrzednych wejscia) pola recepcyjnego
    // potrzebnego do policzenia kafelka wyjsciowego tego work-itemu,
    // z uwzglednieniem stride/paddingu.
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;
    // Ile koncowych/poczatkowych pozycji zbuforowanej linii wejscia wypada
    // poza realnym wejsciem (czyli trzeba je traktowac jako zero-padding),
    // przyciete tak, zeby nigdy nie przekroczyc dlugosci samej linii.
    const int right_unreachable_count_x = min(max(0, input_x + INPUT_LINE_SIZE - input_spatial_size_x),
                                                INPUT_LINE_SIZE);
    const int left_unreachable_count_x = min(max(0, -input_x), INPUT_LINE_SIZE);

    // Wyliczenie offsetow wejscia:
    // Pitch'e (kroki elementow) dla blokowego layoutu wejscia b_fs_yx_fsv16:
    // kolejnosc w pamieci to [batch][slice kanalow][y][x][kanal-w-slice],
    // czyli 16 kanalow jednego slice'a jest wymiarem najszybciej zmiennym
    // (ciaglym) - dokladnie to sprawia, ze 16-elementowe odczyty blokowe
    // i shuffle miedzy lane'ami sa wydajne.
    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_total_f_size = INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM;
    const uint input_b_pitch = input_fs_pitch * ((input_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    // Ile calych slice'ow kanalow "before"-paddingu poprzedza realne kanaly
    // wejscia (potrzebne, bo padding jest wyrazony w kanalach, a
    // adresowanie w pamieci odbywa sie w calych slice'ach).
    const uint input_fs_pad_before = INPUT0_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    // Liniowy offset (w elementach) input[b][slice-kanalow=0][input_y][input_x][kanal=0],
    // czyli bazowy adres, od ktorego ten work-item zacznie czytac swoje
    // pole recepcyjne (slice kanalow wejscia `icb` i kanal-w-slice `sglid`
    // sa dodawane do tego offsetu dalej w kodzie).
    const uint input_offset = b * input_b_pitch +
                              input_fs_pad_before * input_fs_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_Y + input_y) * input_y_pitch +
                              (INPUT0_PAD_BEFORE_SIZE_X + input_x) * input_x_pitch;

    // Wyliczenie offsetow wyjscia:

#if OUTPUT_FORMAT_BFYX
    // Zwykle (nieblokowane) wyjscie bfyx: kolejnosc w pamieci to
    // [batch][kanal][y][x], kanaly NIE sa przeplatane po 16, wiec kanal
    // wyjsciowy tego lane'a (feature_block*16 + sglid) jest po prostu
    // niezalezna plaszczyzna, `feature_block * FEATURE_SLICE_SIZE`
    // slice'ow dalej.
    const uint output_y_pitch = (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * (OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM);

    // Offset output[b][feature_block*16 + sglid][y][x] - zauwaz, ze
    // `output_fs_pitch` dziala tu jak pitch pojedynczej plaszczyzny kanalu,
    // a `sglid` jest dodawany wprost jako offset kanalu (a nie jako
    // rozrzut lane'ow przy odczycie blokowym), bo ta galaz zapisuje
    // skalary, nie 16-elementowe bloki.
    const uint output_offset = b * output_b_pitch +
                               feature_block * (output_fs_pitch * FEATURE_SLICE_SIZE) +
                               (sglid + OUTPUT_PAD_BEFORE_FEATURE_NUM) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X);
#else
    // Blokowe wyjscie b_fs_yx_fsv16: ten sam layout "slice 16 kanalow obok
    // siebie" co wejscie, wiec `output_offset` to baza calego 16-kanalowego
    // slice'a przy pikselu (x, y) - poszczegolne lane'y sa adresowane przez
    // zapisy blokowe dalej w kodzie, a nie przez `+ sglid` tutaj.
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);
    const uint output_total_f_size = OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM;
    const uint output_fs_pitch = output_y_pitch * (OUTPUT_PAD_BEFORE_SIZE_Y + OUTPUT_SIZE_Y + OUTPUT_PAD_AFTER_SIZE_Y);
    const uint output_b_pitch = output_fs_pitch * ((output_total_f_size + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);
    const uint output_fs_pad_before = OUTPUT_PAD_BEFORE_FEATURE_NUM / FEATURE_SLICE_SIZE;

    const uint output_offset = b * output_b_pitch +
                               (feature_block + output_fs_pad_before) * output_fs_pitch +
                               (y + OUTPUT_PAD_BEFORE_SIZE_Y) * output_y_pitch +
                               (x + OUTPUT_PAD_BEFORE_SIZE_X) * output_x_pitch;
#endif

    // Wyliczenie offsetow filtra (wag):
    // Wagi sa przechowywane mniej wiecej jako os_is_yx_isv16_osv16: dla
    // danej pary (slice wyjscia os, slice wejscia is), FILTER_SIZE_X *
    // FILTER_SIZE_Y punktow przestrzennych filtra ma dla kazdego punktu
    // pelny pod-blok 16(wejscie) x 16(wyjscie): `filter_isv_pitch` (=16)
    // przesuwa sie po kanalach wejscia w obrebie slice'a, `filter_x_pitch`/
    // `filter_y_pitch` przesuwaja sie po punktach przestrzennych filtra,
    // a `filter_is_pitch`/`filter_os_pitch` przesuwaja sie po calych
    // slice'ach kanalow wejscia/wyjscia.
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_SIZE_X;
    const uint filter_is_pitch = filter_y_pitch * FILTER_SIZE_Y;
    const uint filter_os_pitch = filter_is_pitch * ((FILTER_IFM_NUM + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

#if BIAS_TERM
    // Inicjalizacja akumulatora biasem kanalu wyjsciowego tego lane'a,
    // rozgloszonym (broadcast) na wszystkie OUTPUT_X_BLOCK_SIZE pozycje
    // kafelka.
#if SLM_DIV_FACTOR == 1
    vec_t dst = (vec_t)(DT_INPUT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
    // Gdy kilka sub-group dzieli miedzy siebie redukcje po kanalach
    // wejscia (SLM_DIV_FACTOR > 1), tylko JEDNA z nich powinna doliczyc
    // bias - inaczej po polaczeniu czesciowych sum ponizej zostalby
    // dodany wielokrotnie. Pozostale startuja od zera.
    vec_t dst;

    if (feature_sub_block == 0) {
        dst = (vec_t)(DT_INPUT_BLOCK_READ(biases, feature_block * FEATURE_SLICE_SIZE));
    } else {
        dst = INPUT0_VAL_ZERO;
    }
#endif // SLM_DIV_FACTOR == 1
#else
    // Brak biasu: akumulator startuje od zera.
    vec_t dst = INPUT0_VAL_ZERO;
#endif // BIAS_TERM

#if SLM_DIV_FACTOR > 1
    // Bufor w pamieci lokalnej (SLM) uzywany do zebrania czesciowych
    // akumulatorow policzonych przez kazda z SLM_DIV_FACTOR sub-group
    // dzielacych ta work-group, zeby mozna je bylo potem zsumowac w
    // ostateczny wynik.
    __local vec_t partial_summ[WORK_GROUP_SIZE];
#endif

#if MULTIPLE_GROUPS_INPUT_PRELOAD
    // --- Sciezka "wiele malutkich grup w jednej sub-group" ------------
    // Uzywana, gdy grupy sa tak male (malo kanalow wejscia/wyjscia
    // kazda), ze kilka calych grup miesci sie naraz w jednej 16-lanowej
    // sub-group. Kazdy lane to para (offset grupy `g`, kanal-wyjscia-
    // -w-obrebie-grupy `ofm_in_group`) zamiast plaskiego indeksu kanalu
    // wyjscia.
    const uint in_split_offset = feature_block * input_fs_pitch;
    const uint g = sglid / (FEATURE_SLICE_SIZE / groups_per_sub_group);
    const uint ofm_in_group = sglid % (FEATURE_SLICE_SIZE / groups_per_sub_group);
    const uint grouped_filter_offset = (group + g) * FILTER_GROUPS_PITCH;
#else
#if GROUPED
    // --- Sciezka "jedna lub kilka grup w jednej sub-group" -------------
    // Petla po kazdej grupie, ktora obejmuje ten 16-kanalowy slice
    // wyjscia (zwykle tylko jedna iteracja, chyba ze slice przechodzi
    // przez granice grup, patrz `groups_per_sub_group` wyzej).
    for (uint g = group; g < group + groups_per_sub_group; g++) {
        const uint in_split_offset = g * input_fs_pitch * (FILTER_IFM_NUM / FEATURE_SLICE_SIZE);
        const uint filter_split_offset = g * FILTER_GROUPS_PITCH;
        const uint filter_offset = (feature_block % (FILTER_OFM_NUM / FEATURE_SLICE_SIZE)) * filter_os_pitch;
#else
        // Zwykla, niegrupowa konwolucja: brak offsetu wejscia/filtra
        // zaleznego od grupy.
        const uint in_split_offset = 0;
        const uint filter_split_offset = 0;
        const uint filter_offset = feature_block * filter_os_pitch;
#endif  // GROUPED
        const uint grouped_filter_offset = filter_offset + filter_split_offset;
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD

        // Ostateczny bazowy offset wejscia dla tej iteracji grupy.
        const uint grouped_input_offset = input_offset + in_split_offset;

        // Zewnetrzna petla redukcji: przechodzi po slice'ach kanalow
        // wejscia (icb), kazdy niosacy 16 kanalow wejscia liczonych razem.
        // Gdy SLM_DIV_FACTOR > 1, ta sub-group obsluguje tylko przypisana
        // sobie czesc zakresu icb; reszte licza siostrzane sub-group,
        // a wyniki sa laczone pozniej przez `partial_summ`.
#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif // SLM_DIV_FACTOR > 1
            // Software prefetch (LSC) dla NASTEPNEGO bloku icb: podpowiadamy
            // sprzetowi zeby zaczal sciagac do cache L1/L3 dane wejscia i wag
            // kolejnej iteracji JUZ TERAZ, zanim faktycznie tam dotrzemy -
            // dzieki temu latencja pamieci globalnej dla icb+1 czesciowo
            // chowa sie za obliczeniami biezacej iteracji icb. To CZYSTA
            // podpowiedz (nic nie czyta do rejestru, nie zmienia wynikow) -
            // ten sam mechanizm co cooperative_prefetch_2d w oneDNN (patrz
            // include na gorze pliku).
            if (icb + 1 < IC_BLOCKS) {
                // Bez opencl_unroll_hint: INPUT_LINE_SIZE / 8 to wyrazenie
                // wyliczane z parametrow generowanych per-instancja kernela
                // i dla malych INPUT_LINE_SIZE (< 8) wychodzi 0 - a
                // opencl_unroll_hint wymaga DODATNIEJ stalej calkowitej,
                // wiec dla takich instancji kompilator (IGC) odrzuca kod
                // bledem "requires a positive integral compile time
                // constant expression". To tylko podpowiedz dla
                // auto-unrollingu, wiec bezpiecznie ja pomijamy.
                // mad24/mul24: 24-bitowe mnozenie calkowitoliczbowe jest
                // tansze na ISA Intel GPU niz pelne 32-bitowe mnozenie.
                // Bezpieczne tutaj - mnozniki (icb, xb) sa zawsze male,
                // a pitche mieszcza sie w 24 bitach dla realistycznych
                // rozmiarow tensora; drugi argument mad24 (akumulator) moze
                // byc dowolnej wielkosci. Baza (icb+1)*input_fs_pitch nie
                // zalezy od xb, wiec liczymy ja raz przed petla zamiast przy
                // kazdej iteracji.
                const uint icb1_input_base = mul24((uint)(icb + 1), input_fs_pitch);
                for (int xb = 0; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                    __builtin_IB_lsc_prefetch_global_uint(
                            (const __global uint*)(input + grouped_input_offset +
                                    mad24((uint)xb, input_x_pitch, icb1_input_base)),
                            0, LSC_LDCC_L1C_L3C);
                }
                __builtin_IB_lsc_prefetch_global_uint(
                        (const __global uint*)(weights + grouped_filter_offset +
                                mul24((uint)(icb + 1), filter_is_pitch)),
                        0, LSC_LDCC_L1C_L3C);
            }

            // Przechodzimy po punktach Y filtra. Rozwiniete (unroll), bo
            // FILTER_SIZE_Y jest stala kompilacyjna, wiec zamienia sie to
            // w kod liniowy bez petli.
            __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
            for (int kh = 0; kh < FILTER_SIZE_Y; kh++) {
                // Pomijamy punkty Y, ktore w calosci wypadaja poza
                // (niedopelnionym) wejsciem - odpowiada to zerowemu
                // wkladowi, ale jest tansze niz buforowanie/mnozenie zer.
                if (input_y + kh * DILATION_SIZE_Y < 0 || input_y + kh * DILATION_SIZE_Y >= INPUT0_SIZE_Y)
                    continue;

                // Baza offsetu wiersza wejscia dla (icb, kh) - nie zalezy od
                // xb, wiec liczymy ja raz na wejscie do petli kh zamiast przy
                // kazdym odczycie xb nizej. mad24: mnozniki (icb) sa male,
                // pitche mieszcza sie w 24 bitach dla realistycznych
                // rozmiarow tensora, a akumulator (grouped_input_offset)
                // moze byc dowolny.
                const uint input_row_base = mad24((uint)icb, input_fs_pitch,
                        grouped_input_offset + (uint)(kh * DILATION_SIZE_Y) * input_y_pitch);

                // Cache jednego wiersza wejscia (jednego punktu Y),
                // wystarczajaco szeroki, zeby pokryc kazda pozycje X
                // wejscia potrzebna dla kazdego punktu kw i kazdej pozycji
                // X wyjscia w kafelku tego work-itemu (INPUT_LINE_SIZE =
                // (OUTPUT_X_BLOCK_SIZE-1)*STRIDE_SIZE_X +
                // (FILTER_SIZE_X-1)*DILATION_SIZE_X + 1, liczone po
                // stronie hosta). Zbuforowanie raz i reuzycie dla kazdego
                // kw unika ponownego czytania tych samych wartosci wejscia
                // FILTER_SIZE_X razy.
                INPUT_TYPE line_cache[INPUT_LINE_SIZE];

#if INPUT_LEFTOVERS
                // Wolniejsza, skalarna sciezka uzywana tylko dla
                // ostatniego, czesciowo wypelnionego slice'a kanalow
                // wejscia (gdy FILTER_IFM_NUM nie jest wielokrotnoscia
                // 16): czyta element po elemencie, zeby kanaly i pozycje
                // X spoza zakresu dalo sie osobno zastapic zerem.
                if ((icb + 1) * FEATURE_SLICE_SIZE >= FILTER_IFM_NUM)
                {
                    for (int xb = 0; xb < INPUT_LINE_SIZE; xb++)
                    {
                        const int in_x = input_x + xb;
                        if (icb * FEATURE_SLICE_SIZE + sglid >= FILTER_IFM_NUM || in_x < 0 || in_x >= input_spatial_size_x)
                            line_cache[xb] = 0;
                        else
                            line_cache[xb] = input[mad24((uint)xb, input_x_pitch, input_row_base) +
                                                   sglid];
                    }
                }
                else
#endif  // INPUT_LEFTOVERS
                {
                    // Szybka sciezka: slice kanalow wejscia jest w calosci
                    // wazny (realne, nie-paddingowe kanaly), wiec mozna
                    // bezpiecznie uzyc szerokich odczytow blokowych. Tylko
                    // padding X na obu koncach linii wymaga jawnego
                    // wyzerowania.
                    int xb = 0;
                    // Wyzerowanie lewego marginesu wypadajacego przed
                    // realnym wejsciem (lewy/gorny zero-padding konwolucji).
                    for (int i = 0; i < left_unreachable_count_x; i++){
                        line_cache[xb + i] = 0;
                    }
                    xb += left_unreachable_count_x;
                    const int reachable_size = INPUT_LINE_SIZE - right_unreachable_count_x;
                    // Wypelnianie "realnego" srodka linii po 8 elementow
                    // naraz szerokim odczytem blokowym sub-group (kazdy z
                    // 16 lane'ow dostaje wartosc swojego kanalu dla kazdej
                    // z 8 kolejnych pozycji X odczytanych w tym wywolaniu).
                    for (; xb + 8 <= reachable_size; xb += 8) {
                        INPUT_TYPE8 vv = DT_INPUT_BLOCK_READ8(input,
                                mad24((uint)xb, input_x_pitch, input_row_base));

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                        line_cache[xb + 4] = vv[4];
                        line_cache[xb + 5] = vv[5];
                        line_cache[xb + 6] = vv[6];
                        line_cache[xb + 7] = vv[7];
                    }
                    // Potem dokanczamy ewentualna reszte po 4.
                    for (; xb + 4 <= reachable_size; xb += 4) {
                        INPUT_TYPE4 vv = DT_INPUT_BLOCK_READ4(input,
                                mad24((uint)xb, input_x_pitch, input_row_base));

                        line_cache[xb + 0] = vv[0];
                        line_cache[xb + 1] = vv[1];
                        line_cache[xb + 2] = vv[2];
                        line_cache[xb + 3] = vv[3];
                    }
                    // I wreszcie ewentualne pojedyncze elementy jeden po drugim.
                    for (; xb < reachable_size; xb++) {
                        line_cache[xb] = DT_INPUT_BLOCK_READ(input,
                                mad24((uint)xb, input_x_pitch, input_row_base));
                    }
                    // Wyzerowanie prawego marginesu wypadajacego poza
                    // realnym wejsciem (prawy/dolny zero-padding).
                    for (int i = 0; i < right_unreachable_count_x; i++){
                        line_cache[xb + i] = 0;
                    }
                }

                // Przechodzimy po punktach X filtra. Tez rozwiniete
                // (unroll) w czasie kompilacji.
                __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
                for (int kw = 0; kw < FILTER_SIZE_X; kw++) {
                    // Zbieramy, dla tego punktu kw, OUTPUT_X_BLOCK_SIZE
                    // wartosci wejscia (jedna na kazda pozycje X wyjscia w
                    // kafelku) z zbuforowanej linii, uwzgledniajac
                    // stride/dilation.
                    vec_t src;
                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
                    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                        // Szybki przypadek specjalny: konwolucja 1x1 ze
                        // stride 1 - i-ty piksel wyjscia mapuje sie
                        // bezposrednio na i-ta zbuforowana wartosc.
                        src[i] = line_cache[i];
#else
                        // Przypadek ogolny: nakladamy offset punktu,
                        // dilation i stride, zeby znalezc wlasciwa
                        // zbuforowana pozycje wejscia.
                        src[i] = line_cache[kw * DILATION_SIZE_X + STRIDE_SIZE_X * i];
#endif  // FILTER_SIZE_X == 1 && DILATION_SIZE_X == 1 && STRIDE_SIZE_X == 1
                    }
#if MULTIPLE_GROUPS_INPUT_PRELOAD
                    // --- MAC dla sciezki "wiele malutkich grup upakowanych
                    // w jednej sub-group": kazdy lane laduje swoj (maly)
                    // wektor wag dla danej grupy jako zwykle skalary (nie
                    // da sie tu zrobic 16-elementowego odczytu blokowego,
                    // bo kanaly wejscia nie wypelniaja calego slice'a na
                    // grupe) i liczy iloczyn skalarny po FILTER_IFM_NUM
                    // kanalach wejscia wprost.
                    typedef MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_IFM_NUM) ifm_vec_t;

                    ifm_vec_t wei0 = FILTER_VAL_ZERO;
                    // Baza (kh, kw) nie zalezy od ifm - liczymy raz przed
                    // petla. mad24: mnozniki (kh, kw, ifm) sa male, pitche
                    // wag sa tanim, malym stalym rozmiarem (FEATURE_SLICE_SIZE
                    // razy male stale filtra), wiec zawsze mieszcza sie
                    // w 24 bitach.
                    const uint filter_khkw_base = mad24((uint)kh, filter_y_pitch,
                            mad24((uint)kw, filter_x_pitch, grouped_filter_offset + ofm_in_group));
                    for (int ifm = 0; ifm < FILTER_IFM_NUM; ifm++)
                        wei0[ifm] = weights[mad24((uint)ifm, filter_isv_pitch, filter_khkw_base)];

                    // Uwaga: `src` w tym miejscu wciaz trzyma JEDEN kanal
                    // wejscia na lane (slice tej grupy, do ktorej nalezy
                    // ten lane); GET_SRC(src, id) rozglasza wartosc z
                    // konkretnego lane'a (konkretnego kanalu wejscia
                    // grupy) do wszystkich lane'ow, zeby dalo sie ja
                    // pomnozyc przez wage tego kanalu dla wlasnego kanalu
                    // wyjscia tego lane'a - ten sam trik z broadcastem, co
                    // w sciezce bez preloadu nizej, tylko po mniejszej
                    // liczbie kanalow.
#if FILTER_IFM_NUM == 2
                        const vec_t src0  = GET_SRC(src, g * FILTER_IFM_NUM + 0);
                        const vec_t src1  = GET_SRC(src, g * FILTER_IFM_NUM + 1);

                        dst = mad(wei0.s0, src0,  dst);
                        dst = mad(wei0.s1, src1,  dst);
#elif FILTER_IFM_NUM == 4
                        const vec_t src0  = GET_SRC(src, g * FILTER_IFM_NUM + 0);
                        const vec_t src1  = GET_SRC(src, g * FILTER_IFM_NUM + 1);
                        const vec_t src2  = GET_SRC(src, g * FILTER_IFM_NUM + 2);
                        const vec_t src3  = GET_SRC(src, g * FILTER_IFM_NUM + 3);

                        dst = mad(wei0.s0, src0,  dst);
                        dst = mad(wei0.s1, src1,  dst);
                        dst = mad(wei0.s2, src2,  dst);
                        dst = mad(wei0.s3, src3,  dst);
#elif FILTER_IFM_NUM == 8
                        const vec_t src0  = GET_SRC(src, g * FILTER_IFM_NUM + 0);
                        const vec_t src1  = GET_SRC(src, g * FILTER_IFM_NUM + 1);
                        const vec_t src2  = GET_SRC(src, g * FILTER_IFM_NUM + 2);
                        const vec_t src3  = GET_SRC(src, g * FILTER_IFM_NUM + 3);
                        const vec_t src4  = GET_SRC(src, g * FILTER_IFM_NUM + 4);
                        const vec_t src5  = GET_SRC(src, g * FILTER_IFM_NUM + 5);
                        const vec_t src6  = GET_SRC(src, g * FILTER_IFM_NUM + 6);
                        const vec_t src7  = GET_SRC(src, g * FILTER_IFM_NUM + 7);

                        dst = mad(wei0.s0, src0,  dst);
                        dst = mad(wei0.s1, src1,  dst);
                        dst = mad(wei0.s2, src2,  dst);
                        dst = mad(wei0.s3, src3,  dst);
                        dst = mad(wei0.s4, src4,  dst);
                        dst = mad(wei0.s5, src5,  dst);
                        dst = mad(wei0.s6, src6,  dst);
                        dst = mad(wei0.s7, src7,  dst);
#else
                        // Ta sciezka preloadu obsluguje tylko 2/4/8
                        // kanalow wejscia na grupe; wszystko inne musi isc
                        // sciezka ogolna (bez preloadu).
#   error convolution_gpu_bfyx_f16.cl: unsupported input feature size for multiple groups input preload
#endif  // FILTER_IFM_NUM
#else
                    // --- MAC dla sciezki ogolnej (jeden pelny 16-kanalowy
                    // slice na sub-group) ------------------------------
                    // Kazdy lane laduje 16 wag laczacych JEGO WLASNY kanal
                    // wyjscia ze wszystkimi 16 kanalami wejscia biezacego
                    // slice'a icb, dla tego punktu (kh, kw) - podzielone na
                    // dwa 8-elementowe odczyty blokowe (wei0 = kanaly
                    // wejscia 0..7, wei1 = kanaly wejscia 8..15).
                    // mad24: mnozniki (icb, kh, kw) sa male, pitche wag sa
                    // malymi stalymi (kombinacje FEATURE_SLICE_SIZE i
                    // wymiarow filtra), zawsze mieszcza sie w 24 bitach;
                    // akumulator (grouped_filter_offset) moze byc dowolny.
                    const uint filter_base = mad24((uint)icb, filter_is_pitch,
                            mad24((uint)kh, filter_y_pitch,
                                    mad24((uint)kw, filter_x_pitch, grouped_filter_offset)));
                    FILTER_TYPE8 wei0 = DT_FILTER_BLOCK_READ8(weights, filter_base);
                    FILTER_TYPE8 wei1 = DT_FILTER_BLOCK_READ8(weights,
                            mad24(8u, filter_isv_pitch, filter_base));
#if GROUPED
                    if (groups_per_sub_group > 1) {
                            // Ten slice przechodzi przez granice grup:
                            // wagi, ktore wlasnie odczytalismy blokowo,
                            // zakladaly jeden ciagly zakres 16 kanalow
                            // wyjscia, ale lane'y nalezace do roznych grup
                            // potrzebuja wag z roznych (niezaleznych)
                            // pod-blokow wag. `correct_lane` mapuje ten
                            // lane na wlasciwa pozycje w bloku wag JEGO
                            // WLASNEJ grupy, a shuffle nizej pobiera
                            // wartosc stamtad.
                            uint correct_lane = sglid % FILTER_OFM_NUM;
                            #if FILTER_TYPE_SIZE == 2
                               short8 w0 = intel_sub_group_shuffle(as_short8(wei0), correct_lane);
                               short8 w1 = intel_sub_group_shuffle(as_short8(wei1), correct_lane);
                               wei0 = AS_FILTER_TYPE8(w0);
                               wei1 = AS_FILTER_TYPE8(w1);
                            #else
                               wei0 = _sub_group_shuffle(wei0, correct_lane);
                               wei1 = _sub_group_shuffle(wei1, correct_lane);
                            #endif

                            // Zerujemy wagi dla lane'ow spoza tej grupy
                            // (realny kanal wyjscia tego lane'a nalezy do
                            // innej grupy niz ta, ktora wlasnie akumulujemy
                            // jako `g`, wiec nie moze dostac zadnego
                            // wkladu z danych wejsciowych tej grupy).
                            if (g != my_group) {
                                wei0 = (FILTER_TYPE8)(0);
                                wei1 = (FILTER_TYPE8)(0);
                            }
                    }
#endif
                    // Rozglaszamy kazdy z 16 kanalow wejscia (jeden na
                    // lane zrodlowy) z `src` do wszystkich lane'ow w
                    // sub-group. Po tym kazdy lane ma wlasna, prywatna
                    // kopie WSZYSTKICH 16 wartosci kanalow wejscia dla
                    // tego punktu (icb, kh, kw), wiec moze niezaleznie
                    // dokonczyc swoj wlasny iloczyn skalarny dla swojego
                    // kanalu wyjscia nizej.
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

                    // Fused multiply-add: akumulujemy
                    // dst[i] += waga[kanal-wyjscia-tego-lane'a][kanal-wejscia] * src[kanal-wejscia][i]
                    // po wszystkich 16 kanalach wejscia biezacego slice'a,
                    // dla kazdej pozycji X wyjscia i w kafelku naraz
                    // (dst/srcN to wektory szerokie na OUTPUT_X_BLOCK_SIZE).
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
#endif  // MULTIPLE_GROUPS_INPUT_PRELOAD
                }
            }
        }
#if GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD
    }
    // koniec petli po grupach `for (uint g = group; ...)` otwartej wyzej
#endif  // GROUPED && !MULTIPLE_GROUPS_INPUT_PRELOAD

#if SLM_DIV_FACTOR > 1
    // Publikujemy czesciowy akumulator tej sub-group (obejmujacy tylko jej
    // ulamek kanalow wejscia) do pamieci lokalnej, a potem czekamy, az
    // kazda sub-group w work-group zrobi to samo.
    partial_summ[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tylko "pierwsza" sub-group kazdego slice'a wyjscia konczy robote:
    // sumuje swoj wlasny czesciowy wynik plus czesciowy wynik kazdej
    // siostrzanej sub-group dla tego samego piksela wyjscia, dopelniajac
    // redukcje po calym zakresie kanalow wejscia.
    if (feature_sub_block == 0) {
        unroll_for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += partial_summ[lid1 % feature_per_wg + i * feature_per_wg];
#endif // SLM_DIV_FACTOR > 1

    // Aplikujemy funkcje aktywacji (np. ReLU) skonfigurowana dla tego kernela.
    dst = ACTIVATION(dst, ACTIVATION_PARAMS);

    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, OUTPUT_X_BLOCK_SIZE) out_vec_t;
    out_vec_t res;

#if OUTPUT_LEFTOVERS
    // Slice kanalow wyjscia tej work-group nie jest w calosci "realny"
    // (czesc z jego 16 lane'ow odpowiadalaby kanalom paddingowym poza
    // OUTPUT_FEATURE_NUM), wiec wracamy do zapisu skalar po skalarze i
    // osobno strzezemy/pomijamy kanaly i pozycje X spoza zakresu.
    if ((feature_block + 1) * FEATURE_SLICE_SIZE >= OUTPUT_FEATURE_NUM) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {

#if HAS_FUSED_OPS
            // Aplikujemy ewentualne fused post-ops (np. zespolony
            // eltwise/quantize) doczepione do tej konwolucji, wariant skalarny.
            FUSED_OPS_SCALAR;
#   if OUTPUT_FORMAT_BFYX
            res[i] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SCALAR);
#   else
            res[i] = FUSED_OPS_RESULT_SCALAR;
#   endif
#else
            res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif

#if OUTPUT_FORMAT_BFYX
            // Straznik: zapisujemy tylko jesli zarowno kanal, jak i
            // pozycja X miesza sie w realnych (niedopelnionych)
            // granicach wyjscia.
            if ((feature_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
                output[output_offset + i] = res[i];
            }
#else
            if ((feature_block * FEATURE_SLICE_SIZE + sglid < OUTPUT_FEATURE_NUM) && (x + i) < OUTPUT_SIZE_X) {
                output[output_offset + i * output_x_pitch + sglid] = res[i];
            }
#endif
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        // Szybka sciezka: caly kafelek wyjscia (wszystkie
        // OUTPUT_X_BLOCK_SIZE pozycji X) miesci sie w calosci w realnych
        // granicach wyjscia, wiec da sie go zapisac jednym szerokim
        // vector store/block-write zamiast zapisow skalarnych.
        if (x + OUTPUT_X_BLOCK_SIZE <= OUTPUT_SIZE_X || OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE == 0) {
#if HAS_FUSED_OPS
            // Wektorowy wariant fused post-ops.
            FUSED_OPS_VEC;
#   if OUTPUT_FORMAT_BFYX
            res = TO_OUTPUTVTYPE(FUSED_OPS_RESULT_VEC);
#   else
            res = FUSED_OPS_RESULT_VEC;
#   endif
#else
#   if OUTPUT_FORMAT_BFYX
            res = TO_OUTPUTVTYPE(dst);
#   else
            res = dst;
#   endif
#endif
            // TODO Generalize for other block sizes
#if OUTPUT_FORMAT_BFYX
            // Zwykle wyjscie bfyx: ciagly vstoreN zapisuje wszystkie
            // OUTPUT_X_BLOCK_SIZE piksele pojedynczego kanalu tego lane'a.
    #if OUTPUT_X_BLOCK_SIZE == 2 || OUTPUT_X_BLOCK_SIZE == 4 || OUTPUT_X_BLOCK_SIZE == 8
            VSTORE(res, 0, output + output_offset);
    #elif OUTPUT_X_BLOCK_SIZE == 1
            output[output_offset] = res[0];
    #else
    #   error convolution_gpu_bfyx_f16.cl: unsupported output x block size
    #endif
#else
            // Blokowe wyjscie fsv16: zapis blokowy sub-group rozrzuca
            // wektor po 16 kanalach-lane'ach dla kazdej pozycji X,
            // produkujac przeplatany-co-16-kanalow layout w pamieci.
    #if OUTPUT_X_BLOCK_SIZE == 8
            DT_OUTPUT_BLOCK_WRITE8(output, output_offset, res);
    #elif OUTPUT_X_BLOCK_SIZE == 4
            DT_OUTPUT_BLOCK_WRITE4(output, output_offset, res);
    #elif OUTPUT_X_BLOCK_SIZE == 2
            DT_OUTPUT_BLOCK_WRITE2(output, output_offset, res);
    #elif OUTPUT_X_BLOCK_SIZE == 1
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, res);
    #else
    #   error convolution_gpu_bfyx_f16.cl: unsupported output x block size
    #endif
#endif  // OUTPUT_FORMAT_BFYX
        } else {
            // Przypadek prawej krawedzi: kafelek tylko czesciowo pokrywa
            // sie z wyjsciem (OUTPUT_SIZE_X nie jest wielokrotnoscia
            // szerokosci kafelka i to jest ostatni kafelek w wierszu),
            // wiec zapisujemy tylko wazne, poczatkowe pozycje, jedna po
            // drugiej.
            for (int i = 0; i < OUTPUT_SIZE_X % OUTPUT_X_BLOCK_SIZE; i++) {
#if HAS_FUSED_OPS
                FUSED_OPS_SCALAR;
#   if OUTPUT_FORMAT_BFYX
                res[i] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SCALAR);
#   else
                res[i] = FUSED_OPS_RESULT_SCALAR;
#   endif
#else
                res[i] = TO_OUTPUT_TYPE(dst[i]);
#endif

#if OUTPUT_FORMAT_BFYX
                output[output_offset + i] = res[i];
#else
                DT_OUTPUT_BLOCK_WRITE(output, output_offset + i * output_x_pitch, res[i]);
#endif
            }
        }
    }
#if SLM_DIV_FACTOR > 1
    // zamyka `if (feature_sub_block == 0)` otwarte wyzej: tylko ta
    // sub-group dociera do tego miejsca i zapisuje wyjscie; pozostale sa
    // gotowe, gdy tylko opublikuja swoja czesciowa sume do `partial_summ`.
    }
#endif
}

// Usuwamy (undef) kazde pomocnicze makro zdefiniowane na gorze tego pliku,
// zeby nie "wycieklo" do tego, co zostanie doklejone (#include) po tym
// kernelu w tej samej jednostce translacji (system budowania
// kernel_selector potrafi sklejac ze soba wiele plikow .cl z kernelami).
#undef AS_INPUT_SRC
#undef AS_US_SRC
#undef GET_SRC
#undef FEATURE_SLICE_SIZE
#undef FILTER_OFM_NUM_ALIGNED
#undef FILTER_IFM_NUM_ALIGNED

#undef INPUT_TYPE
#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8

#undef FILTER_TYPE8

#undef AS_INPUT_TYPE
#undef AS_INPUT_TYPE2
#undef AS_INPUT_TYPE4
#undef AS_INPUT_TYPE8

#undef AS_FILTER_TYPE8

#if OUTPUT_FORMAT_BFYX
#   undef OUTPUTVTYPE
#   undef TO_OUTPUTVTYPE
#   undef VSTORE
#endif  // OUTPUT_FORMAT_BFYX
