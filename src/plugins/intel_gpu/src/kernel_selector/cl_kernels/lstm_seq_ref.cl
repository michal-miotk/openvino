// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

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
    #ifdef SEQUENCE
        INPUT3_TYPE R_copy[NUM_HIDDEN_TO_DO][GATE_NUM][HIDDEN_SIZE];
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
            barrier(CLK_LOCAL_MEM_FENCE);
        #endif
        unroll_for(uint n=0;n<NUM_HIDDEN_TO_DO;++n) { //kernel responsible for HIDDEN_SIZE
            const uint hidden_idx = local_idx*NUM_HIDDEN_TO_DO + n;
            if (hidden_idx >= HIDDEN_SIZE) {
                continue;
            }
            ACCUMULATOR_TYPE gate_output[GATE_NUM];
            unroll_for(uint k=0;k<GATE_NUM;++k){
                INPUT1_TYPE hidden_result = 0;
                const uint weight_idx = hidden_idx+weight_offsets[k];
                #ifdef SEQUENCE
                    if(i==1){
                        unroll_for(uint j=0;j<HIDDEN_SIZE;++j) {
                            R_copy[n][k][j] = R[INPUT3_GET_INDEX(0, weight_idx, j, 0)];
                        }
                    }

                    unroll_for(uint j=0;j<HIDDEN_SIZE;++j) {
                        if(i>0){
                            hidden_result += hidden_history[OUTPUT_GET_INDEX(b, 0, prev_idx, j)]*R_copy[n][k][j];
                        }
                    }
                #endif //SEQUENCE
                if(i==0){
                    #ifdef SEQUENCE
                        //printf("SEQinput sizes %d %d %d %d \n", INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X);
                        hidden_result = initial_hidden_stateR[INPUT1_GET_INDEX(b, weight_idx, 0, 0)];
                        //printf("%d setting hidden b |%d idx %d calc idx to %d initial cell is  %f R is %f hidden_result %f\n", 0, b, weight_idx, INPUT1_GET_INDEX(b, weight_idx, 0, 0), initial_cell_state[INPUT2_GET_INDEX(b, 0, hidden_idx, 0)],  R[INPUT3_GET_INDEX(0, weight_idx, 0, 0)], hidden_result);
                    #else
                        hidden_result = initial_hidden_stateR[INPUT1_GET_INDEX(b, weight_idx, 0, 0)];
                        printf("NON SEQinput sizes %d %d %d %d \n", INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X);
                    #endif
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
            ACCUMULATOR_TYPE temp_cell_state;
            if (i==0){
                #ifdef SEQUENCE
                    temp_cell_state = gate_output[0]*initial_cell_state[INPUT2_GET_INDEX(b, 0, hidden_idx, 0)] + gate_output[1]*gate_output[2];
                #else
                    temp_cell_state = gate_output[0]*initial_cell_state[INPUT2_GET_INDEX(b, hidden_idx, 0, 0)] + gate_output[1]*gate_output[2];
                #endif
            }else{
                temp_cell_state *= gate_output[0];
                temp_cell_state += gate_output[1]*gate_output[2];
            }
            
            #if DIRECTION == 1  //reverse
                const uint cur_history_idx = real_seq_length - 1 - i ;
            #else
                const uint cur_history_idx = i;
            #endif
            #ifdef SEQUENCE
                hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state, ACTIVATION_PARAMS_H);
                printf("I will write idden_state to batch %d idx %d %f \n", b, OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0), hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)]);
            #else
                hidden_state[OUTPUT_GET_INDEX(b, hidden_idx, 0, 0)] = gate_output[3]*ACTIVATION_H(temp_cell_state, ACTIVATION_PARAMS_H);
            #endif
            #ifdef SEQUENCE
                printf("I will write to batch %d idx %d %f \n", b, OUTPUT_GET_INDEX(b, 0, cur_history_idx, hidden_idx), hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)]);
                hidden_history[OUTPUT_GET_INDEX(b, 0, cur_history_idx, hidden_idx)] = hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 0)];
            #endif
            if(i==real_seq_length-1){
                #ifdef SEQUENCE
                    cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 0)] = temp_cell_state;
                    printf("I will write cell to batch %d idx %d %f \n", b, OUTPUT2_GET_INDEX(b, 0, hidden_idx, 0), temp_cell_state);
                #else
                    cell_state[OUTPUT1_GET_INDEX(b, hidden_idx, 0, 0)] = temp_cell_state;
                #endif
            }
        }
    }   
}
