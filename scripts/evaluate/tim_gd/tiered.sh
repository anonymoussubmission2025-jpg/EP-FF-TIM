# ===========================> WRN 28-10 <=========================================

python3 -m src.main \
		-F logs/tim_gd/tiered/wideres \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/wrn28_10" \
		dataset.split_dir="split/tiered" \
		model.arch='wrn28_10' \
		model.num_classes=351 \
		tim.iter=1000 \
		evaluate=True \
		eval.method='tim_gd'