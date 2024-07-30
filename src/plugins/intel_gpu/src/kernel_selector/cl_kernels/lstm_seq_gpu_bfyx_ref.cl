// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* x,
    const __global INPUT1_TYPE* initial_hidden_state,
    const __global INPUT2_TYPE* initial_cell_state,
    const __global INPUT3_TYPE* sequence_lengths,
    const __global INPUT4_TYPE* W,
    const __global INPUT5_TYPE* R,
    const __global INPUT6_TYPE* B,
    __global OUTPUT_TYPE* hidden_history,
    __global OUTPUT1_TYPE* cell_state
)
{
    const uint hidden_idx = get_global_id(0);
    float local_hidden_state = 0;
    const uint b = get_global_id(1);
    //
    const int weight_offsets[4] = {GEMM_OFFSET_F, GEMM_OFFSET_I, GEMM_OFFSET_Z, GEMM_OFFSET_O};
    const int gate_num = 4;
    float hidden_result[gate_num];
    float input_result[gate_num];
    float gate_output[gate_num];
    for(int k=0;k<gate_num;k++){
        gate_output[k] = 0;
    }
    pinrtf("W is %f R is %f B is %f\n, W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[0], 0, 0)], R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[0],  0, 0)], B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[0], 0, 0)]);
    for(int i=0;i<MAX_SEQ_LENGTH;i++){
        for(int k=0;k<gate_num;k++){
            hidden_result[k] = 0;
            input_result[k] = 0;
        }
        for(int k=0;k<gate_num;k++){
            for(int j=0;j<HIDDEN_SIZE;j++) {
                if(i==0){
                    hidden_result[k] += initial_hidden_state[INPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)]*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k],  j, 0)];
                }else{
                    hidden_result[k] += local_hidden_state*R[INPUT5_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
                }
            }
            
            for(int j=0;j<INPUT_SIZE;j++) {
                input_result[k] += x[INPUT0_GET_INDEX_SAFE(b, i, j, 0)]*W[INPUT4_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], j, 0)];
            }
            gate_output[k] = hidden_result[k] + input_result[k] + B[INPUT6_GET_INDEX_SAFE(0, hidden_idx+weight_offsets[k], 0, 0)];
        
            switch(k){
                case 0:
                case 1:
                case 3:
                    gate_output[k] = ACTIVATION_F(ACTIVATION_CLIP(gate_output[k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_F);
                    break;
                case 2:
                    gate_output[k] = ACTIVATION_G(ACTIVATION_CLIP(gate_output[k], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_G);
                    break;
                default:
                    break;
            }
        }

        if (i==0){
            cell_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] = gate_output[0]*initial_cell_state[INPUT2_GET_INDEX_SAFE(b, 0, hidden_idx, 0)];
            cell_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] += gate_output[1]*gate_output[2];
        }else{
            cell_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] *= gate_output[0];
            cell_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)] += gate_output[1]*gate_output[2];
        }
        local_hidden_state = gate_output[3]*ACTIVATION_H(ACTIVATION_CLIP(cell_state[OUTPUT1_GET_INDEX_SAFE(b, 0, hidden_idx, 0)], ACTIVATION_PARAMS_CLIP), ACTIVATION_PARAMS_H);
        hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, i, hidden_idx)] = local_hidden_state;
    }
    printf("R is %p B is %p ; hidden history %p cell state %p batch %d\n", &R[0], &B[0], &hidden_history[0],  &cell_state[0], b);
    printf("result is %f %f \n", hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, 0, 0)], hidden_history[OUTPUT_GET_INDEX_SAFE(b, 0, 1, 0)]);
}
