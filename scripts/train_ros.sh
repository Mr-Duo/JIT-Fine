#!/bin/bash

setup=("SETUP1" "SETUP2" "SETUP3" "SETUP4" "SETUP5")
sampling=("ros")

# Nested loop
echo "Nested loop processing:"
for i in "${sampling[@]}"; do
    for j in "${setup[@]}"; do    
        echo "Train $k $j with $i"
        mkdir -p $j/$i/dg_cache/save/FFmpeg/jitfine/checkpoints

        python -m JITFine.concat.run \
            --output_dir=$j/$i/dg_cache/save/FFmpeg/jitfine/checkpoints \
            --config_name=microsoft/codebert-base \
            --model_name_or_path=microsoft/codebert-base \
            --tokenizer_name=microsoft/codebert-base \
            --do_train \
            --train_data_file dataset/FFmpeg/$j/$i/$j-FFmpeg-deepjit-train.jsonl dataset/FFmpeg/$j/$i/$j-FFmpeg-features-train.jsonl \
            --eval_data_file dataset/FFmpeg/$j/$j-FFmpeg-deepjit-val.jsonl dataset/FFmpeg/$j/$j-FFmpeg-features-val.jsonl\
            --test_data_file dataset/FFmpeg/$j/$j-FFmpeg-deepjit-test.jsonl dataset/FFmpeg/$j/$j-FFmpeg-features-test.jsonl\
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
            --seed 42 2>&1| tee $j/$i/dg_cache/save/FFmpeg/jitfine/train.log \
            --cache_dir cache_ros


        python -m JITFine.concat.run \
            --output_dir=$j/$i/dg_cache/save/FFmpeg/jitfine/checkpoints \
            --config_name=microsoft/codebert-base \
            --model_name_or_path=microsoft/codebert-base \
            --tokenizer_name=microsoft/codebert-base \
            --do_test \
            --train_data_file dataset/FFmpeg/$j/$i/$j-FFmpeg-deepjit-train.jsonl dataset/FFmpeg/$j/$i/$j-FFmpeg-features-train.jsonl \
            --eval_data_file dataset/FFmpeg/$j/$j-FFmpeg-deepjit-val.jsonl dataset/FFmpeg/$j/$j-FFmpeg-features-val.jsonl\
            --test_data_file dataset/FFmpeg/$j/$j-FFmpeg-deepjit-test.jsonl dataset/FFmpeg/$j/$j-FFmpeg-features-test.jsonl\
            --epoch 50 \
            --max_seq_length 512 \
            --max_msg_length 64 \
            --train_batch_size 256 \
            --eval_batch_size 25 \
            --learning_rate 2e-5 \
            --max_grad_norm 1.0 \
            --evaluate_during_training \
            --only_adds \
            --buggy_line_filepath=$j/$i/dg_cache/save/FFmpeg/jitfine/changes_complete_buggy_line_level.pkl \
            --seed 42 2>&1 | tee $j/$i/dg_cache/save/FFmpeg/jitfine/test.log \
            --cache_dir cache_ros
    done
done
