# # ===========================> TIM-GD <=========================================
python3 -m src.main \
		-F logs/tim_gd/mini_10_ways/wrn28_10 \
		with dataset.path="data/mini_imagenet" \
		tim.iter=1000 \
		ckpt_path="checkpoints/mini/softmax/wrn28_10" \
		dataset.split_dir="split/mini" \
		model.arch='wrn28_10' \
		evaluate=True \
		eval.method="tim_gd" \
		eval.n_ways=10

python3 -m src.main \
		-F logs/tim_gd/mini_20_ways/wrn28_10 \
		with dataset.path="data/mini_imagenet" \
		tim.iter=1000 \
		ckpt_path="checkpoints/mini/softmax/wrn28_10" \
		dataset.split_dir="split/mini" \
		model.arch='wrn28_10' \
		evaluate=True \
		eval.method="tim_gd" \
		eval.n_ways=20