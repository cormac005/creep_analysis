DATA_RAW     := data/raw/CreepData.xlsx
TRIMMED      := data/processed/trimmed_experiment.h5
FIT_RESULTS  := data/processed/tlv_fit_results.h5
EDA_RESULTS  := data/processed/eda_results.h5

$(TRIMMED): $(DATA_RAW) src/creep_model/config.py scripts/pipeline/01_classify_and_trim.py
	python scripts/pipeline/01_classify_and_trim.py

$(FIT_RESULTS): $(TRIMMED) scripts/pipeline/02_fit_tlv.py
	python scripts/pipeline/02_fit_tlv.py

$(EDA_RESULTS): $(TRIMMED) scripts/pipeline/03_compute_eda_stats.py
	python scripts/pipeline/03_compute_eda_stats.py

figures: $(FIT_RESULTS) $(EDA_RESULTS)
	python scripts/pipeline/04_generate_tlv_plots.py
	python scripts/pipeline/05_generate_eda_plots.py

.PHONY: refit
refit:
	touch scripts/pipeline/02_fit_tlv.py   # forces the expensive stage even if inputs look unchanged
	$(MAKE) $(FIT_RESULTS)

all: figures