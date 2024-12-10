#!/bin/bash

setup=("SETUP1" "SETUP2")
sampling=("unsampling")
project="FFmpeg"

# Nested loop
echo "Nested loop processing:"
for i in "${sampling[@]}"; do
    for j in "${setup[@]}"; do    
        echo "Train $k $j with $i"
        mkdir -p PDBert/$j/$i/dg_cache/save/$project/jitfine/checkpoints

        python -m JITFine.concat.run \
            --output_dir=PDBert/$j/$i/dg_cache/save/$project/jitfine/checkpoints \
            --cache_dir=PDBert/$j/$i/dg_cache/save/$project/jitfine/checkpoints \
            --model_name_or_path=../pretrain/PDBert/data/models/pdbert-base \
            --do_train \
            --train_data_file dataset/$project/$j/$i/$j-$project-deepjit-train.jsonl dataset/$project/$j/$i/$j-$project-features-train.jsonl \
            --eval_data_file dataset/$project/$j/$j-$project-deepjit-val.jsonl dataset/$project/$j/$j-$project-features-val.jsonl\
            --test_data_file dataset/$project/$j/$j-$project-deepjit-test.jsonl dataset/$project/$j/$j-$project-features-test.jsonl\
            --epoch 50 \
            --max_seq_length 512 \
            --max_msg_length 64 \
            --train_batch_size 24 \
            --eval_batch_size 128 \
            --learning_rate 1e-5 \
            --max_grad_norm 1.0 \
            --evaluate_during_training \
            --feature_size 14 \
            --patience 10 \
            --seed 42 2>&1| tee $j/$i/dg_cache/save/$project/jitfine/train.log \
            # --config_name=microsoft/codebert-base \
            # --tokenizer_name=microsoft/codebert-base \


        python -m JITFine.concat.run \
            --output_dir=PDBert/$j/$i/dg_cache/save/$project/jitfine/checkpoints \
            --cache_dir=PDBert/$j/$i/dg_cache/save/$project/jitfine/checkpoints \
            --model_name_or_path=../pretrain/PDBert/data/models/pdbert-base \
            --do_test \
            --train_data_file dataset/$project/$j/$i/$j-$project-deepjit-train.jsonl dataset/$project/$j/$i/$j-$project-features-train.jsonl \
            --eval_data_file dataset/$project/$j/$j-$project-deepjit-val.jsonl dataset/$project/$j/$j-$project-features-val.jsonl\
            --test_data_file dataset/$project/$j/$j-$project-deepjit-test.jsonl dataset/$project/$j/$j-$project-features-test.jsonl\
            --epoch 50 \
            --max_seq_length 512 \
            --max_msg_length 64 \
            --train_batch_size 256 \
            --eval_batch_size 25 \
            --learning_rate 2e-5 \
            --max_grad_norm 1.0 \
            --evaluate_during_training \
            --only_adds \
            --buggy_line_filepath=$j/$i/dg_cache/save/$project/jitfine/changes_complete_buggy_line_level.pkl \
            --seed 42 2>&1 | tee $j/$i/dg_cache/save/$project/jitfine/test.log \
            # --config_name=microsoft/codebert-base \
            # --tokenizer_name=microsoft/codebert-base \
    done
done
