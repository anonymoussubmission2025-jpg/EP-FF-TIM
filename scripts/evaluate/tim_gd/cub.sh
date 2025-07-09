# ===========================> WRN 28-10 <=====================================
python3 -m src.main \
		-F logs/tim_gd/cub/ \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/wrn28_10" \
		dataset.split_dir="split/cub" \
		model.arch='wrn28_10' \
		model.num_classes=200 \
		tim.iter=1000 \
		evaluate=True \
		eval.method='tim_gd'