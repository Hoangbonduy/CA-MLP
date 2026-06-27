from pathlib import Path

import matplotlib.pyplot as plt


PRED_LEN = 96
LOOK_BACK_WINDOWS = [48, 96, 192, 336, 512, 720]
METRICS = [("mse", "MSE"), ("mae", "MAE")]
MODEL_ORDER = ["CA-MLP", "TimeMixer", "iTransformer", "DLinear"]
MODEL_STYLE = {
	"CA-MLP": {"marker": "o", "linewidth": 2.4},
	"TimeMixer": {"marker": "s", "linewidth": 2.0},
	"iTransformer": {"marker": "^", "linewidth": 2.0},
	"DLinear": {"marker": "D", "linewidth": 2.0},
}


DATA = {
	"ETTh2": {
		"CA-MLP": {
			48: {"mse":0.3000534176826477, "mae":0.3442128896713257},
			96: {"mse":0.2938849627971649, "mae":0.3393665552139282},
			192: {"mse":0.2884504497051239, "mae":0.3399693965911865},
			336: {"mse":0.28223392367362976, "mae":0.33954736590385437},
			512: {"mse":0.27482885122299194, "mae":0.33311814069747925},
			720: {"mse":0.27434128522872925, "mae":0.3373395502567291},
		},
		"TimeMixer": {
			48: {"mse":0.3009280860424042, "mae":0.3452633321285248},
			96: {"mse":0.28924083709716797, "mae":0.3423728346824646},
			192: {"mse":0.2946704030036926, "mae":0.3490089476108551},
			336: {"mse":0.2899320423603058, "mae":0.35021713376045227},
			512: {"mse":0.2942974865436554, "mae":0.3539731502532959},
			720: {"mse":0.3017708957195282, "mae":0.3600127398967743},
		},
		"iTransformer": {
			48: {"mse": 0.3093286156654358, "mae": 0.35151758790016174},
			96: {"mse": 0.3004031479358673, "mae": 0.34961849451065063},
			192: {"mse": 0.30209484696388245, "mae": 0.355439156293869},
			336: {"mse": 0.3070763647556305, "mae": 0.36315521597862244},
			512: {"mse": 0.30375349521636963, "mae": 0.3594362735748291},
			720: {"mse": 0.30318230390548706, "mae": 0.3641250729560852},
		},
		"DLinear": {
			48: {"mse": 0.39627543091773987, "mae": 0.4372069537639618},
			96: {"mse": 0.3414474129676819, "mae": 0.3952932059764862},
			192: {"mse": 0.3234822452068329, "mae": 0.38183826208114624},
			336: {"mse": 0.30734774470329285, "mae": 0.36979350447654724},
			512: {"mse": 0.3025066554546356, "mae": 0.3675801753997803},
			720: {"mse": 0.30420058965682983, "mae": 0.37025386095046997},
		},
		
	},
	"ETTm2": {
		"CA-MLP": {
			48: {"mse": 0.19086243212223053, "mae": 0.27676844596862793},
			96: {"mse": 0.17845045030117035, "mae": 0.2634199261665344},
			192: {"mse": 0.16929887235164642, "mae": 0.2526433765888214},
			336: {"mse": 0.16389015316963196, "mae": 0.2533254325389862},
			512: {"mse": 0.1633429378271103, "mae": 0.2521003186702728},
			720: {"mse": 0.16224515438079834, "mae": 0.25253263115882874},
		},
		"iTransformer": {
			48: {"mse": 0.19096744060516357, "mae": 0.27397292852401733},
			96: {"mse": 0.18378707766532898, "mae": 0.26686403155326843},
			192: {"mse": 0.1857031285762787, "mae": 0.26994407176971436},
			336: {"mse": 0.17371851205825806, "mae": 0.2651403546333313},
			512: {"mse": 0.18013902008533478, "mae": 0.27182239294052124},
			720: {"mse": 0.18020053207874298, "mae": 0.2734571099281311},
		},
		"TimeMixer": {
			48: {"mse": 0.18653273582458496, "mae": 0.26998743414878845},
			96: {"mse": 0.17498546838760376, "mae": 0.258059561252594},
			192: {"mse": 0.1721295416355133, "mae": 0.25743040442466736},
			336: {"mse": 0.17819856107234955, "mae": 0.26756078004837036},
			512: {"mse": 0.17466597259044647, "mae": 0.265739768743515},
			720: {"mse": 0.1918584257364273, "mae": 0.28575587272644043},
		},
		"DLinear": {
			48: {"mse": 0.2111673355102539, "mae": 0.31234610080718994},
			96: {"mse": 0.19342845678329468, "mae": 0.2928171455860138},
			192: {"mse": 0.1790960282087326, "mae": 0.2738305330276489},
			336: {"mse": 0.16939815878868103, "mae": 0.26572561264038086},
			512: {"mse": 0.1675756722688675, "mae": 0.26328301429748535},
			720: {"mse": 0.16244257986545563, "mae": 0.2562496066093445},
		},
	},
}


def plot_dataset_metric(dataset_name, dataset_values, metric_key, metric_label, output_dir):
	fig, ax = plt.subplots(figsize=(7.2, 4.8))

	for model_name in MODEL_ORDER:
		model_values = dataset_values.get(model_name, {})
		x = [window for window in LOOK_BACK_WINDOWS if window in model_values]
		if not x:
			continue

		y = [model_values[window][metric_key] for window in x]
		style = MODEL_STYLE.get(model_name, {})
		ax.plot(x, y, label=model_name, **style)

	ax.set_title(f"{dataset_name} - {metric_label} (pred_len={PRED_LEN})")
	ax.set_xlabel("Look Back Window")
	ax.set_ylabel(metric_label)
	ax.set_xticks(LOOK_BACK_WINDOWS)
	ax.grid(True, linestyle="--", alpha=0.35)

	handles, labels = ax.get_legend_handles_labels()
	if handles:
		ax.legend(loc="best", frameon=False)

	fig.tight_layout()

	output_path = output_dir / f"{dataset_name.lower()}_{metric_key}_look_back_window.png"
	fig.savefig(output_path, dpi=300)
	plt.close(fig)
	print(f"Saved: {output_path}")


def main():
	output_dir = Path(__file__).resolve().parents[1] / "experiment_figures"
	output_dir.mkdir(parents=True, exist_ok=True)

	for dataset_name, dataset_values in DATA.items():
		for metric_key, metric_label in METRICS:
			plot_dataset_metric(dataset_name, dataset_values, metric_key, metric_label, output_dir)


if __name__ == "__main__":
	main()
