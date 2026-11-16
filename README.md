# battery_pack_inconsistency_quantitative_evaluation
Graph-based quantitative evaluation of lithium-ion battery pack inconsistency using a global consistency baseline
Use `Data preprocessing` to process the open-source dataset available at [https://ivstskl.changan.com.cn/?p=2697](https://ivstskl.changan.com.cn/?p=2697), in compliance with its terms of use, to obtain preprocessed data.
Then use `get_Drop`, `get_DTW`, and `get_R0(Rint-ER-VFRLS)` to extract inconsistency features.
Next, perform Range80 Standardization on the extracted features.
Finally, feed the standardized features into `Evaluation of battery group inconsistency` to obtain the evaluation results of battery pack inconsistency.
Figures and further analysis can be implemented according to your own requirements.
