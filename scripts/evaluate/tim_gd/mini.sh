# ===========================> WRN 28-10 <=====================================

python3 -m src.main \
		-F logs/tim_gd/mini/wideres \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/wrn28_10" \
		dataset.split_dir="split/mini" \
		model.arch='wrn28_10' \
		evaluate=True \
		tim.iter=1000 \
		eval.method='tim_gd'