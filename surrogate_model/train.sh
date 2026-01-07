# Run with 1 GPU
level=0
seed='2025'
DOMAIN=Weather
DIR=tear/surrogate_model
DATA_PATH=${DIR}/surrogate_data/${DOMAIN}_${level}
BART_PATH=''
SAVE_PATH=${DIR}/${DOMAIN}_${level}_${seed}

export CUDA_VISIBLE_DEVICES=0

if [ -d "$SAVE_PATH" ]; then 
    echo "Directory '$SAVE_PATH' already exists"
else
    # Attempt to create directory
    if mkdir -p "$SAVE_PATH"; then
        echo "Created directory: $SAVE_PATH"
    else
        echo "Error: Failed to create directory '$SAVE_PATH'" >&2
        exit 1
    fi
fi


TOTAL_NUM_UPDATES="8000"
WARMUP_UPDATES=400
LR=1e-4
MAX_TOKENS="4096"
UPDATE_FREQ="1"
size="large"

# train
mkdir -p ${SAVE_PATH}
python custom_train.py --num-workers 16 ${DATA_PATH}/bins \
    --user-dir src/ \
    --seed ${seed} --save-dir ${SAVE_PATH} \
    --keep-best-checkpoints 3\
    --restore-file ${BART_PATH}/model.pt \
    --max-tokens $MAX_TOKENS \
    --task text_to_table_task  --table-max-columns 20 \
    --source-lang text --target-lang data \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_ours_$size --return-relative-column-strs col_head  \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr $LR --max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --warmup-init-lr '1e-07' \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters 2>&1 | tee ${SAVE_PATH}/log


d=${SAVE_PATH}
n=3
ls $d/*pt -lht
ckpts=`ls $d/checkpoint*best_*pt -lht | tail -n $n | rev | cut -d " " -f1 | rev`
echo $ckpts
python $DIR/average_checkpoints.py --inputs $ckpts --output $d/checkpoint_average_best-${n}.pt
