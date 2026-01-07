seed=2025
level=3
DOMAIN=Incidents
DIR=/export/data/tlice/schema_data/tear/surrogate_model
DATA_PATH=${DIR}/surrogate_data/${DOMAIN}_${level}
ckpt=${DIR}/${DOMAIN}_${level}_${seed}/checkpoint_average_best-3.pt

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

fairseq-interactive ${DATA_PATH}/bins --path $ckpt --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 8192 --max-len-b 1024 --user-dir src/ --task text_to_table_task  --table-max-columns 20 > $ckpt.valid_constrained.out < ${DATA_PATH}/valid.bpe.text
fairseq-interactive ${DATA_PATH}/bins --path $ckpt --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 8192 --max-len-b 1024 --user-dir src/ --task text_to_table_task  --table-max-columns 20 > $ckpt.test_constrained.out < ${DATA_PATH}/test.bpe.text

bash convert_fairseq_output_to_text.sh $ckpt.valid_constrained.out
bash convert_fairseq_output_to_text.sh $ckpt.test_constrained.out