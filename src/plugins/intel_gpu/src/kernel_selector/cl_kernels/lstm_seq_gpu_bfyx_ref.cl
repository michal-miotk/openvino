// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// initial_hidden_state
// initial_cell_state     
// sequence_lengths
// WR
// B
// output0  
//output1
//output2
KERNEL(lstm_seq)(
    const __global INPUT0_TYPE* x,
    const __global INPUT0_TYPE* initial_hidden_state,
    const __global INPUT0_TYPE* initial_cell_state,
    const __global INPUT0_TYPE* WxplusB,
    const __global INPUT0_TYPE* R,
    const __global INPUT0_TYPE* B,
    __global OUTPUT_TYPE* hidden_history,
    __global OUTPUT_TYPE* output_value_of_hidden_state,
    __global OUTPUT_TYPE* cell_state
)
{
    const uint hidden_idx = get_global_id(0);
    const uint b = get_global_id(1);
    global ACCUMULATOR[BATCH_SIZE][HIDDEN_SIZE][4] hidden_result;
    global ACCUMULATOR[BATCH_SIZE][HIDDEN_SIZE] input_result;
    global ACCUMULATOR[BATCH_SIZE][HIDDEN_SIZE] forget_gate_output;
    for(int i=0;i<SEQ_LENGTH;i++){
        //if( i == 0){
            for(int j=0;j<HIDDEN_SIZE;j++) {
                for(int z=0;z<4;z++)
                hidden_result[b][hidden_idx][z] += initial_hidden_state[INPUT1_GET_INDEX(b, hidden_idx, 0, 0)]*R[INPUT4_GET_INDEX(0, j, hidden_idx+z*HIDDEN_SIZE, 0)];
            }
            for(int j=0;j<INPUT_SIZE;j++) {
                input_result[b][hidden_idx] += x[INPUT0_GET_INDEX(b, hidden_idx, j)]*W[INPUT3_GET_INDEX(0, hidden_idx, j, 0)]
            }
            for(int j=0;j<HIDDEN_SIZE;j++){
                forget_gate_output[b][hidden_idx] += hidden_result[b][j] + input_result[b][j] + B[INPUT5_GET_INDEX(0, hidden_idx, 0, 0)];
            }
            cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 1)] = cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 1)] * ACTIVATION_KERNEL(forget_gate_output[b][hidden_idx], ACTIVATION_PARAMS_F);
            cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 1)] += ACTIVATION_KERNEL(forget_gate_output[b][hidden_idx], ACTIVATION_PARAMS_G)*ACTIVATION_KERNEL(forget_gate_output[b][hidden_idx], ACTIVATION_PARAMS_H);
            output_value_of_hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 1)] = ACTIVATION_KERNEL(cell_state[OUTPUT2_GET_INDEX(b, 0, hidden_idx, 1)], ACTIVATION_PARAMS_H) * 
            hidden_history[OUTPUT0_GET_INDEX(b, 0, i, hidden_idx)] = output_value_of_hidden_state[OUTPUT1_GET_INDEX(b, 0, hidden_idx, 1)];
        //}
    }
}
