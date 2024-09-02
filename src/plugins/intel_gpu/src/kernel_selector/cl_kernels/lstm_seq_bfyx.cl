// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"

#define INPUT0_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define INPUT1_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT1_TYPE, VEC_SIZE)
#define INPUT3_TYPE_VEC  MAKE_VECTOR_TYPE(INPUT3_TYPE, VEC_SIZE)
#define OUTPUT_TYPE_VEC  MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define READ_VEC(offset, ptr) CAT(vload, VEC_SIZE)(offset, ptr)

KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* xWB,
    const __global INPUT1_TYPE* initial_hidden_stateR,
    const __global INPUT2_TYPE* initial_cell_state,
    const __global INPUT3_TYPE* R,
#ifdef SEQUENCE
    const __global INPUT4_TYPE* sequence_lengths,
    __global OUTPUT_TYPE* hidden_history,
    __global OUTPUT1_TYPE* hidden_state,
    __global OUTPUT2_TYPE* cell_state
#else
    __global OUTPUT_TYPE* hidden_state,
    __global OUTPUT1_TYPE* cell_state
#endif
)
{
    __local INPUT1_TYPE_VEC hiddencache[HBLOCK_NUM];
    #ifdef SEQUENCE 
        #if MAX_SEQ_LEN > 1
            INPUT3_TYPE_VEC r_block[NUM_HIDDEN_TO_DO][GATE_NUM][HBLOCK_NUM];
        #endif
    #endif    
    const uint b = get_global_id(1);
    const uint local_idx = get_local_id(0);
    const uint weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O};
    #ifdef SEQUENCE
        const uint real_seq_length = sequence_lengths[INPUT4_GET_INDEX(b, 0, 0, 0)];
    #else
        const uint real_seq_length = 1;
    #endif
    unroll_for(uint i=0;i<real_seq_length;++i){
        #ifdef SEQUENCE
            #if DIRECTION == 1 //reverse
                const uint prev_idx = real_seq_length - i;
            #else
                const uint prev_idx = i-1;
            #endif
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);
        unroll_for(uint l=0;l<NUM_HIDDEN_TO_DO;++l) { //kernel responsible for HIDDEN_SIZE
            const uint hidden_idx = local_idx*NUM_HIDDEN_TO_DO + l;
            if (hidden_idx >= HIDDEN_SIZE) {
                continue;
            }
            if(hidden_idx % VEC_SIZE == 0)
            {
                #ifdef SEQUENCE
                    if(i==0){
                        hiddencache[hidden_idx / VEC_SIZE] = READ_VEC(0, &initial_hidden_state[INPUT1_GET_INDEX(b, 0, hidden_idx, 0)]);
                    } else {
                        hiddencache[hidden_idx / VEC_SIZE] = READ_VEC(0, &hidden_history[OUTPUT_GET_INDEX(b, 0, prev_idx, hidden_idx)]);
                    }
                #else
                    hiddencache[hidden_idx / VEC_SIZE] = READ_VEC(0, &initial_hidden_state[INPUT1_GET_INDEX(b, hidden_idx, 0)]);
                #endif
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        ACCUMULATOR_TYPE temp_cell_state[NUM_HIDDEN_TO_DO];
        unroll_for(uint l=0;l<NUM_HIDDEN_TO_DO;++l) { //kernel responsible for HIDDEN_SIZE
            const uint hidden_idx = local_idx*NUM_HIDDEN_TO_DO + l;
            if (hidden_idx >= HIDDEN_SIZE) {
                continue;
            }
            ACCUMULATOR_TYPE gate_output[GATE_NUM];
            unroll_for(uint k=0;k<GATE_NUM;++k){
                INPUT3_TYPE hidden_result = 0;
                const uint weight_idx = hidden_idx+weight_offsets[k];
                #ifdef SEQUENCE
                    uint rindex = INPUT3_GET_INDEX(0, weight_idx, 0, 0);
                #else
                    uint rindex = INPUT3_GET_INDEX(weight_idx, 0, 0, 0);
                #endif
                unroll_for(uint j=0;j<HBLOCK_NUM;++j) {
                    if(i==0){
                        #ifdef SEQUENCE
                            #if MAX_SEQ_LEN > 1
                                r_block[l][k][j] = READ_VEC(0, &R[rindex]);
                            #else 
                                INPUT3_TYPE_VEC r_block = READ_VEC(0, &R[rindex]); 
                            #endif
                            rindex += VEC_SIZE;
                            #if MAX_SEQ_LEN > 1
                                hidden_result += dot(hiddencache[j], r_block[l][k][j]);
                            #else
                                hidden_result += dot(hiddencache[j], r_block);
                            #endif
                        #else
                            INPUT3_TYPE_VEC r_block = READ_VEC(0, &R[r_index]);
                            r_index += VEC_SIZE;
                            hidden_result += dot(hiddencache[j], r_block);

                        #endif
                    }else{
                        #ifdef SEQUENCE
                            #if MAX_SEQ_LEN > 1
                                hidden_result += dot(hiddencache[j], r_block[l][k][j]);
                            #endif
                        #endif
                    }
                }
                #if DIRECTION == 1 //reverse
                    gate_output[k] = hidden_result + xWB[INPUT0_GET_INDEX(b, real_seq_length-1-i, weight_idx, 0)];
                #else
                    #ifdef SEQUENCE
                        gate_output[k] = hidden_result + xWB[INPUT0_GET_INDEX(b, i, weight_idx, 0)];
                    #else
                        gate_output[k] = hidden_result + xWB[INPUT0_GET_INDEX(b, weight_idx, 0, 0)];
                    #endif
                #endif //DIRECTION
                switch(k){
                    case 0:
                    case 1:
                    case 3:
                        gate_output[k] = ACTIVATION_F(ACTIVATION_CLIP(TO_OUTPUT_TYPE(gate_output[k]), ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
                        break;
                    case 2:
                        gate_output[k] = ACTIVATION_G(ACTIVATION_CLIP(TO_OUTPUT_TYPE(gate_output[k]), ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);
                        break;
                    default:
                        break;
                }
            }
            if (i==0){
                #ifdef SEQUENCE
                    temp_cell_state[l] = gate_output[0]*initial_cell_state[INPUT2_GET_INDEX(b, 0, hidden_idx, 0)] + gate_output[1]*gate_output[2];
                #else
                    temp_cell_state[l] = gate_output[0]*initial_cell_state[INPUT2_GET_INDEX(b, hidden_idx, 0, 0)] + gate_output[1]*gate_output[2];
                #endif
            }else{
                temp_cell_state[l] *= gate_output[0];
                temp_cell_state[l] += gate_output[1]*gate_output[2];
            }
            
            #if DIRECTION == 1  //reverse
                const uint cur_history_idx = real_seq_length - 1 - i ;
            #else
                const uint cur_history_idx = i;
            #endif
            if(i==real_seq_length-1){
                #ifdef SEQUENCE
                    OUTPUT1_TYPE temp_hidden_state = gate_output[3]*ACTIVATION_H(temp_cell_state[l], ACTIVATION_PARAMS_H);
                    hidden_history[OUTPUT_GET_INDEX(b, 0, cur_history_idx, hidden_idx)] = temp_hidden_state;
                    hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)] = temp_hidden_state;
                    cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 0)] = temp_cell_state[l];
                #else
                    hidden_state[OUTPUT_GET_INDEX(b, hidden_idx, 0, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state[l], ACTIVATION_PARAMS_H);
                    cell_state[OUTPUT1_GET_INDEX(b, hidden_idx, 0, 0)] = temp_cell_state[l];
                #endif
            } else {
                #ifdef SEQUENCE
                    hidden_history[OUTPUT_GET_INDEX(b, 0, cur_history_idx, hidden_idx)] = gate_output[3]*ACTIVATION_H(temp_cell_state[l], ACTIVATION_PARAMS_H);
                #endif
            }
        }
    }   
}
