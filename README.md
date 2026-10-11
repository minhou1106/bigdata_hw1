# Assignment 1 SVD and PiSSA experiments

Student: 陳旻濠  
Student ID: 315552049

This repository contains exactly three files:

| File | Purpose |
|---|---|
| [finetune_q1-q3_final version.ipynb](finetune_q1-q3.ipynb) | Consolidated Q1–Q3 source based on the course notebook, with rank settings, training-loop timing and component selection. |
| [Q4final version.ipynb](finetune_q4.ipynb) | The completed Q4 notebook with its authentic saved execution outputs and all six figures. |
| README.md | Experimental setup, recorded results, observations, provenance and retained source records. |

## Execution status and provenance

The consolidated Q1–Q3 source has been syntax checked and its component-selection logic checked, but it has not been rerun end to end after integration. The tables below come from earlier actual experiments and a later rank16 timing supplement. They are historical results, not newly generated outputs of the consolidated source.

The Q4 notebook is a copy of the student's executed `Q4fianl version.ipynb`, with its filename normalized. Its six embedded plots and saved outputs are preserved. CSV/JSON source contents for Q1–Q3 are included below so separate result files are unnecessary. Original execution notebooks remain in the student's local archive. Source SHA256 values identify the files from which these records were extracted.

## Experimental setup

- Model: meta-llama/Llama-3.2-1B.
- Fine-tuning data: MetaMathQA, 25,000 selected examples and 24,975 retained after tokenization filtering in saved runs.
- Available source settings: seed 42, FP16, one epoch, learning rate 2e-5, sequence length 512, micro batch size 8, gradient accumulation 16 and effective batch size 128.
- Target projections: q, k, v, o, gate, up and down.
- Evaluation: 1,000 examples for each of GSM8K and MATH, with up to 256 generated tokens. Recorded accuracy depends on both generated answers and the retained parsing/equivalence functions.
- The later timing run used two Tesla T4 GPUs and PyTorch 2.11.0+cu128. Exact environment versions and runtime conditions were not retained for every earlier run.

## Q1 Accuracy by rank

| Model | GSM8K accuracy | MATH accuracy |
|---|---:|---:|
| Original | 3.5% | 0.1% |
| Default rank 16 | 15.6% | 3.9% |
| Default rank 64 | 20.4% | 4.7% |
| Default rank 256 | 27.2% | 5.3% |

Accuracy increases with rank in these recorded runs. Relative to rank16, rank256 gains 11.6 percentage points on GSM8K and 1.4 on MATH. Greater adapter capacity may help, but single runs do not establish causality or statistical significance. The original MATH run has 956 unparsed answers and rank16 has 542, so performance also reflects output formatting and extraction.

Sources: `benchmark_results_v1.csv` for original/r16, `64 256.csv` for r64/r256.

## Q2 Training time

| Rank | Training seconds | Training minutes | Trainable parameters |
|---|---:|---:|---:|
| 16 | 3862.044324109 | 64.36740540181667 | 11,272,192 |
| 64 | 4189 | 69.81666666666666 | 45,088,768 |
| 256 | 5657 | 94.28333333333333 | 180,355,072 |

Seconds for ranks64/256 convert the recorded durations 1:09:49 and 1:34:17. Rank16 is a later training-only supplement, measured over the training loop and excluding initialization, evaluation and model saving. Its final loss is 0.45467856526374817 and it completed 196 optimizer updates. It did not rerun GSM8K/MATH evaluation.

Rank64 takes about 5.45 minutes longer than rank16, while rank256 takes about 29.92 minutes longer. Rank256 exceeds rank64 by 24.47 minutes (35.0%). Larger adapters add computation and memory work. Separate sessions limit direct comparisons, and the earlier timing boundaries are not independently verified beyond the retained records.

Sources: `q2_training_times.csv`, `rank16_default_timing.json`, `rank16_default_training_history.csv`, and `64 256.csv`.

## Q3 Component selection at rank16

| Selection | GSM8K | MATH | Final training loss | Initialization seconds |
|---|---:|---:|---:|---:|
| Top16 default | 15.6% | 3.8% | 0.454686 | 144.594209 |
| 25% position | 12.6% | 3.1% | 0.494626 | 151.014038 |
| 50% position | 11.6% | 2.0% | 0.496411 | 151.664388 |
| Bottom16 | 11.5% | 2.6% | 0.506507 | 150.802819 |

All four saved runs use 24,975 training examples, rank16, 11,272,192 trainable parameters, seed42 and 1,000 evaluation examples per task. The merged model is evaluated after merging 112 adapted layers. The 25% and 50% positions refer to singular-value indices, not cumulative energy thresholds.

Top16 performs best on both tasks. It exceeds Bottom16 by 4.1 percentage points on GSM8K and 1.2 on MATH. MATH is not strictly monotonic: Bottom16 (2.6%) exceeds the 50% selection (2.0%). Dominant components may provide a more useful adaptation subspace in this setup, but repeated-seed evidence is unavailable.

Q3 Top16 MATH is 38/1,000 (3.8%), while Q1 rank16 is a separate run with 39/1,000 (3.9%). Both are retained. The recovered Q3 CSV includes reliably saved visible fields and does not claim to recover columns hidden by notebook display truncation.

Sources: `Q3_benchmark_results_recovered.csv` and saved outputs from `q3test.ipynb`.

## Q4 Original query projection spectra

| Layer | Minimum rank for at least 50% energy |
|---|---:|
| 0 | 62 |
| 7 | 120 |
| 15 | 84 |

Cumulative energy uses squared singular values, normalized by the complete spectrum. Each layer has a magnitude bar chart and cumulative-energy line chart for the first200 values. The six original charts are embedded in the Q4 notebook. Raw singular-value arrays are not retained separately.

Layer0 reaches half the energy with the fewest components, while layer7 needs the most. These spectra describe weight energy and do not directly specify the adaptation rank required for an accuracy target.

## Running the source notebooks

Use Kaggle with the intended GPU configuration, enable Internet for downloads, and configure `HF_TOKEN` as a Kaggle secret with model access. Never place the actual token in source or saved outputs. Run cells in order. Core packages are torch, transformers, datasets, accelerate, huggingface_hub, safetensors, numpy, pandas, tqdm and matplotlib. Package versions are not pinned because complete historical versions are unavailable.

The Q1–Q3 full run plan evaluates the original model and runs default ranks16/64/256 plus rank16 selections25%/50%/bottom. A new default rank16 run serves both Q1 and Q3. The full plan is expensive and is not required merely to preserve the historical results listed here. Q4 runs separately without fine-tuning or task evaluation.

The code extends the instructor's assignment notebook structure. Course repository: https://github.com/nycubigdata-2026fall/BigData2026-Assignment1. The course materials identify PiSSA (Meng et al., NeurIPS2024) as the original method and source: https://github.com/MuLabPKU/PiSSA.

## Submission

Use a public repository under the student's own GitHub account. Include its URL in the English report, then export the completed report from Word as an A4 PDF of at most three pages, named `Asg1_315552049.pdf`, and submit it to E3. This package has not been uploaded or submitted by the assistant.

## Retained source records

The following collapsible sections contain exact CSV/JSON contents, including available fields omitted from the summary tables. These are recorded data, not generated experiment outputs.

<details>
<summary>benchmark_results_v1.csv</summary>

SHA256: `4b48a033e8eb27171b8c74c9189096fbbbc08d9996acb58e4c526d647004878c`

```csv
run_tag,training_dataset,training_examples,method,rank,component,gsm8k_accuracy,gsm8k_correct,gsm8k_total,gsm8k_unparsed,gsm8k_eval_seconds,math_accuracy,math_correct,math_total,math_unparsed,math_eval_seconds,init_seconds,trainable_parameters,status,seed,merge_seconds,merged_layer_count,evaluated_after_merge,full_model_saved,full_model_path,final_training_loss,training_history_path,total_parameters_with_adapter,max_pissa_reconstruction_error
metamath_fresh,meta-math/MetaMathQA,0,fresh,,,0.035,35,1000,15,1476.3202620809998,0.001,1,1000,956,1481.7778867480001,0.0,0,ok,42,,,,,,,,,
metamath_pissa_default_r16,meta-math/MetaMathQA,24975,pissa,16.0,default,0.156,156,1000,0,1039.5604008249993,0.039,39,1000,542,1460.449523923,142.05876617400008,11272192,ok,42,0.16330100200048037,112.0,True,True,/kaggle/working/Assignment1/metamath_pissa_default_r16_full_model,0.4547428637742996,/kaggle/working/Assignment1/metamath_pissa_default_r16_training_history.csv,1247086592.0,0.00020509955356828868
```

</details>

<details>
<summary>64 256.csv</summary>

SHA256: `f2cb14e91df41ed23a0b01291c48be5dc78744fbb42e57f1bf590e62c11931f0`

```csv
rank,component,training_time,training_minutes,gsm8k_accuracy,math_accuracy,trainable_parameters,final_training_loss,init_seconds,status
64,default,1:09:49,69.81666666666666,0.204,0.047,45088768,0.413985,139.439939,ok
256,default,1:34:17,94.28333333333333,0.272,0.053,180355072,0.375353,142.533305,ok
```

</details>

<details>
<summary>Q3_benchmark_results_recovered.csv</summary>

SHA256: `951df2372b2e93415d191c77395d874e40e82e63ac3d647e4b79190134780ec7`

```csv
run_tag,training_dataset,training_examples,method,rank,component,gsm8k_accuracy,math_accuracy,gsm8k_correct,math_correct,init_seconds,final_training_loss,max_pissa_reconstruction_error,gsm8k_total,math_total,trainable_parameters,status,seed,merged_layer_count,evaluated_after_merge,full_model_saved,total_parameters_with_adapter
metamath_pissa_default_r16,meta-math/MetaMathQA,24975,pissa,16,default,0.156,0.038,156,38,144.594209,0.454686,0.000205,1000,1000,11272192,ok,42,112,True,True,1247086592
metamath_pissa_25%_r16,meta-math/MetaMathQA,24975,pissa,16,25%,0.126,0.031,126,31,151.014038,0.494626,0.000207,1000,1000,11272192,ok,42,112,True,True,1247086592
metamath_pissa_50%_r16,meta-math/MetaMathQA,24975,pissa,16,50%,0.116,0.02,116,20,151.664388,0.496411,0.000207,1000,1000,11272192,ok,42,112,True,True,1247086592
metamath_pissa_bottom_r16,meta-math/MetaMathQA,24975,pissa,16,bottom,0.115,0.026,115,26,150.802819,0.506507,0.000207,1000,1000,11272192,ok,42,112,True,True,1247086592
```

</details>

<details>
<summary>q2_training_times.csv</summary>

SHA256: `73a444828157debcbc46b8e09071fc913d4a10eed7ee5864daadd6dc441f2e52`

```csv
rank,component,training_seconds,training_minutes,measurement,training_examples,epochs,learning_rate,seed,micro_batch_size,gradient_accumulation_steps,data_parallel,gpu_count,gpu_names,torch_version,model_id,trainable_parameters,final_training_loss,status
16,default,3862.044324109,64.36740540181667,training_loop_only_new_run,24975,1,2e-05,42,8,16,True,2,"['Tesla T4', 'Tesla T4']",2.11.0+cu128,meta-llama/Llama-3.2-1B,11272192,0.45467856526374817,training_complete_evaluation_not_run
```

</details>

<details>
<summary>rank16_default_timing.json</summary>

SHA256: `a646eca3f006d08be76a82ba6e1ed30ec9b0323e9f2a8350422841b58bfb9b1a`

```json
{
  "rank": 16,
  "component": "default",
  "training_seconds": 3862.044324109,
  "training_minutes": 64.36740540181667,
  "measurement": "training_loop_only_new_run",
  "training_examples": 24975,
  "epochs": 1,
  "learning_rate": 2e-05,
  "seed": 42,
  "micro_batch_size": 8,
  "gradient_accumulation_steps": 16,
  "data_parallel": true,
  "gpu_count": 2,
  "gpu_names": [
    "Tesla T4",
    "Tesla T4"
  ],
  "torch_version": "2.11.0+cu128",
  "model_id": "meta-llama/Llama-3.2-1B",
  "trainable_parameters": 11272192,
  "final_training_loss": 0.45467856526374817,
  "status": "training_complete_evaluation_not_run"
}
```

</details>

<details>
<summary>rank16_default_training_history.csv</summary>

SHA256: `ad02db596f49bbbe1f0c6c9c187e4b877bd9fb9bfcfc4d11534b89af41d5b2ab`

```csv
update,loss,lr
1,0.9178439676761627,6.666666666666667e-06
2,0.9445282705128193,1e-05
3,0.8753638863563538,1.3333333333333333e-05
4,0.8382301777601242,1.6666666666666667e-05
5,0.7943544536828995,2e-05
6,0.7885166481137276,2e-05
7,0.7164125367999077,1.9998633049924693e-05
8,0.6839917004108429,1.999453257340926e-05
9,0.6417461857199669,1.998769969148305e-05
10,0.6511191353201866,1.9978136272187745e-05
11,0.6241052374243736,1.99658449300667e-05
12,0.6197443045675755,1.9950829025450116e-05
13,0.6044142171740532,1.9933092663536384e-05
14,0.557806845754385,1.9912640693269754e-05
15,0.5354929696768522,1.9889478706014687e-05
16,0.5773362591862679,1.9863613034027224e-05
17,0.5303857941180468,1.9835050748723826e-05
18,0.5300411693751812,1.9803799658748096e-05
19,0.4906729571521282,1.9769868307835996e-05
20,0.5472563281655312,1.973326597248006e-05
21,0.5248311832547188,1.9694002659393306e-05
22,0.53781883046031,1.9652089102773487e-05
23,0.5159696843475103,1.9607536761368484e-05
24,0.5276755876839161,1.9560357815343577e-05
25,0.5153191722929478,1.9510565162951538e-05
26,0.4980937670916319,1.9458172417006347e-05
27,0.46590795181691647,1.9403193901161614e-05
28,0.5037032458931208,1.934564464599461e-05
29,0.510875478386879,1.9285540384897073e-05
30,0.4893893916159868,1.922289754977385e-05
31,0.4676985386759043,1.9157733266550577e-05
32,0.48761027306318283,1.909006535049163e-05
33,0.47521185129880905,1.9019912301329593e-05
34,0.48108356073498726,1.8947293298207637e-05
35,0.464867539703846,1.887222819443612e-05
36,0.47478654235601425,1.879473751206489e-05
37,0.46860517375171185,1.8714842436272774e-05
38,0.49612728878855705,1.863256480957574e-05
39,0.4656849093735218,1.854792712585539e-05
40,0.45777277648448944,1.8460952524209355e-05
41,0.48970338329672813,1.8371664782625287e-05
42,0.44381555914878845,1.8280088311480203e-05
43,0.47915796004235744,1.8186248146866928e-05
44,0.4564164113253355,1.8090169943749477e-05
45,0.49381148628890514,1.7991879968949248e-05
46,0.43060220032930374,1.789140509396394e-05
47,0.4466246496886015,1.7788772787621126e-05
48,0.457498487085104,1.7684011108568593e-05
49,0.4815884493291378,1.757714869760335e-05
50,0.40547424368560314,1.7468214769841542e-05
51,0.4943546038120985,1.735723910673132e-05
52,0.4594627469778061,1.7244252047910893e-05
53,0.4259800799190998,1.7129284482913973e-05
54,0.4521100278943777,1.7012367842724887e-05
55,0.45522700995206833,1.6893534091185658e-05
56,0.45635339617729187,1.6772815716257414e-05
57,0.459407240152359,1.6650245721138483e-05
58,0.41975693218410015,1.6525857615241686e-05
59,0.429819842800498,1.6399685405033168e-05
60,0.42646583542227745,1.6271763584735373e-05
61,0.4336730744689703,1.6142127126896682e-05
62,0.44086690433323383,1.6010811472830253e-05
63,0.4403770472854376,1.5877852522924733e-05
64,0.4193534702062607,1.5743286626829437e-05
65,0.45105282962322235,1.560715057351673e-05
66,0.4079768769443035,1.5469481581224274e-05
67,0.43599986657500267,1.533031728727994e-05
68,0.42508697882294655,1.5189695737812153e-05
69,0.4359996747225523,1.504765537734844e-05
70,0.4333491697907448,1.4904235038305084e-05
71,0.455952076241374,1.4759473930370738e-05
72,0.4786134958267212,1.461341162978688e-05
73,0.45920784026384354,1.4466088068528068e-05
74,0.3932778239250183,1.4317543523384928e-05
75,0.4464622139930725,1.4167818604952906e-05
76,0.4127790331840515,1.4016954246529697e-05
77,0.4338000826537609,1.3864991692924524e-05
78,0.43996901996433735,1.3711972489182208e-05
79,0.4094828963279724,1.3557938469225167e-05
80,0.4374037943780422,1.3402931744416432e-05
81,0.40329043939709663,1.3246994692046837e-05
82,0.4317396320402622,1.3090169943749475e-05
83,0.4345707409083843,1.293250037384465e-05
84,0.4175623506307602,1.2774029087618448e-05
85,0.40540784411132336,1.26147994095382e-05
86,0.4096251502633095,1.2454854871407993e-05
87,0.44603688083589077,1.2294239200467516e-05
88,0.41059568524360657,1.213299630743747e-05
89,0.4140502046793699,1.1971170274514802e-05
90,0.4016240984201431,1.1808805343321102e-05
91,0.45573725923895836,1.164594590280734e-05
92,0.42205298133194447,1.148263647711842e-05
93,0.40244678407907486,1.1318921713420691e-05
94,0.42349102534353733,1.1154846369695864e-05
95,0.4339891467243433,1.099045530250463e-05
96,0.4259837754070759,1.0825793454723325e-05
97,0.4534906093031168,1.0660905843256995e-05
98,0.4170617088675499,1.0495837546732224e-05
99,0.43509734235703945,1.0330633693173083e-05
100,0.41958977840840816,1.0165339447663586e-05
101,0.4061415512114763,1e-05
102,0.4089450463652611,9.834660552336415e-06
103,0.4252263307571411,9.669366306826919e-06
104,0.42813795804977417,9.504162453267776e-06
105,0.38739478029310703,9.339094156743007e-06
106,0.4151340890675783,9.174206545276678e-06
107,0.4254828840494156,9.009544697495373e-06
108,0.37829283624887466,8.84515363030414e-06
109,0.4244275987148285,8.68107828657931e-06
110,0.4074447825551033,8.51736352288158e-06
111,0.41119213216006756,8.35405409719266e-06
112,0.3970882724970579,8.191194656678905e-06
113,0.39907933957874775,8.0288297254852e-06
114,0.39463143050670624,7.867003692562533e-06
115,0.4174403380602598,7.705760799532485e-06
116,0.421236177906394,7.545145128592009e-06
117,0.41895901784300804,7.385200590461803e-06
118,0.39934508688747883,7.225970912381557e-06
119,0.43513418547809124,7.067499626155354e-06
120,0.4199572652578354,6.909830056250527e-06
121,0.4609046224504709,6.7530053079531664e-06
122,0.4287992939352989,6.59706825558357e-06
123,0.42481641843914986,6.442061530774835e-06
124,0.4001121036708355,6.2880275108177915e-06
125,0.4165357630699873,6.13500830707548e-06
126,0.4206412509083748,5.983045753470308e-06
127,0.43489311821758747,5.832181395047099e-06
128,0.41793375462293625,5.6824564766150724e-06
129,0.41710325330495834,5.533911931471936e-06
130,0.4213136415928602,5.386588370213124e-06
131,0.39133578538894653,5.240526069629265e-06
132,0.41371254064142704,5.095764961694923e-06
133,0.40583257004618645,4.952344622651566e-06
134,0.4189300797879696,4.8103042621878515e-06
135,0.446363877505064,4.669682712720065e-06
136,0.4113402124494314,4.530518418775734e-06
137,0.4204541239887476,4.392849426483275e-06
138,0.4192060325294733,4.256713373170565e-06
139,0.3998766951262951,4.12214747707527e-06
140,0.4079830553382635,3.989188527169749e-06
141,0.4075491577386856,3.857872873103322e-06
142,0.41890862584114075,3.72823641526463e-06
143,0.4152179453521967,3.6003145949668338e-06
144,0.4223505984991789,3.4741423847583134e-06
145,0.4238121323287487,3.349754278861517e-06
146,0.4111965671181679,3.2271842837425917e-06
147,0.39677703380584717,3.1064659088143424e-06
148,0.40634636394679546,2.9876321572751143e-06
149,0.4296118710190058,2.8707155170860303e-06
150,0.4114084579050541,2.7557479520891104e-06
151,0.3971431441605091,2.642760893268684e-06
152,0.41521192342042923,2.5317852301584642e-06
153,0.40767799504101276,2.422851302396655e-06
154,0.45033440738916397,2.315988891431412e-06
155,0.4342112150043249,2.211227212378877e-06
156,0.4095030650496483,2.1085949060360654e-06
157,0.42246792279183865,2.008120031050753e-06
158,0.4109796956181526,1.9098300562505266e-06
159,0.415572352707386,1.8137518531330768e-06
160,0.42002542316913605,1.7199116885197996e-06
161,0.42900756001472473,1.6283352173747148e-06
162,0.41682301834225655,1.5390474757906449e-06
163,0.40226423740386963,1.4520728741446087e-06
164,0.41827210038900375,1.367435190424261e-06
165,0.4063365273177624,1.2851575637272262e-06
166,0.39800343103706837,1.2052624879351105e-06
167,0.42351013608276844,1.127771805563882e-06
168,0.4007662758231163,1.0527067017923654e-06
169,0.4006728380918503,9.800876986704111e-07
170,0.41988855227828026,9.09934649508375e-07
171,0.400240171700716,8.42266733449425e-07
172,0.3947664871811867,7.771024502261526e-07
173,0.4009413979947567,7.144596151029304e-07
174,0.4327626097947359,6.543553540053926e-07
175,0.4044653847813606,5.968060988383884e-07
176,0.41655435785651207,5.418275829936537e-07
177,0.41357728838920593,4.894348370484648e-07
178,0.39459102414548397,4.396421846564236e-07
179,0.37580554001033306,3.924632386315186e-07
180,0.4223568122833967,3.4791089722651437e-07
181,0.4093328807502985,3.059973406066963e-07
182,0.4262911416590214,2.667340275199426e-07
183,0.43887241743505,2.3013169216400732e-07
184,0.4115576483309269,1.9620034125190645e-07
185,0.3836288694292307,1.6494925127617632e-07
186,0.42133009620010853,1.3638696597277678e-07
187,0.4087430853396654,1.1052129398531508e-07
188,0.42408922873437405,8.735930673024806e-08
189,0.4019917007535696,6.690733646361858e-08
190,0.41139455884695053,4.9170974549885844e-08
191,0.4189728833734989,3.4155069933301535e-08
192,0.41409948840737343,2.1863727812254653e-08
193,0.40119449980556965,1.230030851695263e-08
194,0.4467439241707325,5.467426590739511e-09
195,0.4045561123639345,1.3669500753099586e-09
196,0.45467856526374817,0.0
```

</details>

<details>
<summary>q4_minimum_ranks.csv</summary>

SHA256: `67cfe4d10acb9075e7edb13605803445f0ddb48ed2c14dd4019998b91354095d`

```csv
layer,min_rank_50_percent
0,62
7,120
15,84
```

</details>

<details>
<summary>Retained Q3 notebook table snapshots</summary>

These are exact text/plain display outputs from the saved notebook. Ellipses represent original display truncation and are not filled in.

Source SHA256: `b4a46eceee4492d1225108541603152da9f9967a63b2099829a4ff850629b0fa`

```text
Original cell index 17 (zero-based)
model.safetensors: reconstructing file:   0%|          |  0.00B / 2.47GB            

Original cell index 17 (zero-based)
model.safetensors: downloading bytes:           |  0.00B            

Original cell index 17 (zero-based)
Loading weights:   0%|          | 0/146 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
generation_config.json:   0%|          | 0.00/185 [00:00<?, ?B/s]

Original cell index 17 (zero-based)
Train metamath epoch 1/1:   0%|          | 0/3122 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval gsm8k:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval math:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Loading weights:   0%|          | 0/146 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Train metamath epoch 1/1:   0%|          | 0/3122 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval gsm8k:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval math:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Loading weights:   0%|          | 0/146 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Train metamath epoch 1/1:   0%|          | 0/3122 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval gsm8k:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval math:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Loading weights:   0%|          | 0/146 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Train metamath epoch 1/1:   0%|          | 0/3122 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval gsm8k:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Eval math:   0%|          | 0/250 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]

Original cell index 17 (zero-based)
                      run_tag      training_dataset  training_examples method  \
0  metamath_pissa_default_r16  meta-math/MetaMathQA              24975  pissa   
1      metamath_pissa_25%_r16  meta-math/MetaMathQA              24975  pissa   
2      metamath_pissa_50%_r16  meta-math/MetaMathQA              24975  pissa   
3   metamath_pissa_bottom_r16  meta-math/MetaMathQA              24975  pissa   

   rank component  init_seconds  trainable_parameters status  seed  ...  \
0    16   default    144.594209              11272192     ok    42  ...   
1    16       25%    151.014038              11272192     ok    42  ...   
2    16       50%    151.664388              11272192     ok    42  ...   
3    16    bottom    150.802819              11272192     ok    42  ...   

   math_eval_seconds  merge_seconds  merged_layer_count  \
0        1469.272072       0.157241                 112   
1        1481.621810       0.147078                 112   
2        1487.311600       0.147539                 112   
3        1472.856544       0.147238                 112   

   evaluated_after_merge  full_model_saved  \
0                   True              True   
1                   True              True   
2                   True              True   
3                   True              True   

                                     full_model_path  final_training_loss  \
0  /kaggle/working/Assignment1/metamath_pissa_def...             0.454686   
1  /kaggle/working/Assignment1/metamath_pissa_25%...             0.494626   
2  /kaggle/working/Assignment1/metamath_pissa_50%...             0.496411   
3  /kaggle/working/Assignment1/metamath_pissa_bot...             0.506507   

                               training_history_path  \
0  /kaggle/working/Assignment1/metamath_pissa_def...   
1  /kaggle/working/Assignment1/metamath_pissa_25%...   
2  /kaggle/working/Assignment1/metamath_pissa_50%...   
3  /kaggle/working/Assignment1/metamath_pissa_bot...   

   total_parameters_with_adapter  max_pissa_reconstruction_error  
0                     1247086592                        0.000205  
1                     1247086592                        0.000207  
2                     1247086592                        0.000207  
3                     1247086592                        0.000207  

[4 rows x 29 columns]

Original cell index 18 (zero-based)
  method  rank component  gsm8k_accuracy  math_accuracy  gsm8k_correct  \
1  pissa    16       25%           0.126          0.031            126   
2  pissa    16       50%           0.116          0.020            116   
3  pissa    16    bottom           0.115          0.026            115   
0  pissa    16   default           0.156          0.038            156   

   gsm8k_total  math_correct  math_total  init_seconds  trainable_parameters  \
1         1000            31        1000    151.014038              11272192   
2         1000            20        1000    151.664388              11272192   
3         1000            26        1000    150.802819              11272192   
0         1000            38        1000    144.594209              11272192   

   gsm8k_unparsed  math_unparsed  
1               0            584  
2               0            610  
3               0            602  
0               0            550  

Original cell index 18 (zero-based)
          gsm8k_accuracy                       math_accuracy               \
component            25%    50% bottom default           25%   50% bottom   
rank                                                                        
16                 0.126  0.116  0.115   0.156         0.031  0.02  0.026   

                   
component default  
rank               
16          0.038  
```

</details>

## Notebook file integrity

| Packaged notebook | SHA256 |
|---|---|
| finetune_q1-q3.ipynb | `cd45228727cd307fc9d4d77d86093bda3e83a6c9dcc7d83a666996d1a4f85e73` |
| finetune_q4.ipynb | `7266e22d42372d67de2275dfd6c56aa8625ababdf2c1769253f905202c428808` |
