# ===========================> WRN 28-10 <=====================================
python3 -m src.main \
		-F logs/tim_gd/mini2cub/wrn28_10 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini2cub/softmax/wrn28_10" \
		dataset.split_dir="split/mini" \
		model.arch='wrn28_10' \
		tim.iter=1000 \
		evaluate=True \
		eval.method='tim_gd' \
		eval.target_data_path="data/cub/CUB_200_2011/images" \
		eval.target_split_dir="split/cub"