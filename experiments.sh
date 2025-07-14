######################################## EXPERIMENT 1 ########################################
# Diferent initialization in various incremental learning approaches.
##############################################################################################

datasets=("FBinc-S" "FBinc-M" "FBinc-L")
inits=(1)
RNS=(0 0.1)
models=("LKGE" "finetune" "incDE" "EWC" "EMR")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for init in "${inits[@]}"; do
      for RN in "${RNS[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init 1 -RN "$RN"
      done
    done
  done
done

datasets=("FBinc-S" "FBinc-M" "FBinc-L")
inits=(0 3)
models=("LKGE" "finetune" "incDE" "EWC" "EMR")

for dataset in "${datasets[@]}"; do
  for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
          # Run the Python script with the current combination of parameters
          python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$model"
      done
  done
done

######################################## EXPERIMENT 2 ########################################
# Number of training epochs.
##############################################################################################

datasets=("FBinc-S" "FBinc-M" "FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150)
models=("finetune" "EWC" "EMR" "LKGE" "incDE")
inits=(0 1 3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch"
      done
    done
  done
done

