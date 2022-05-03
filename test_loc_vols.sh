loc_ranges=("5e6 5e8" "1e7 5e8" "1e7 5e8" "1.5e7 5e8")
readout_p10s=( "0.00" )

folder=$(timedatectl | grep Local | awk '{print "runs_"$4 "_"$5}' | sed 's/-/_/g' | sed 's/:/_/g')
mkdir $folder
mkdir $folder/sim_events
mkdir $folder/plot_data
#folder=$(timedatectl | grep Local | awk '{print "runs_"$4 "_"$5}')
timedatectl > $folder.log

for lr in "${loc_ranges[@]}"; do
    for i in {1..20}; do
	    #echo "running: make_landscapes.py --folder $folder --sampling=sobol --n_points=500 --depolarize $d --prob01 $r"
            out_name=$(echo "posterior_$lr $i.txt" | sed 's/ /_/g')
	    echo "i=$i:"
            echo "\tpython3 gen_gals.py --out-directory $folder/sim_events --volume-range lr"
            echo "\tpython3 est_hubble.py --in-directory $folder/sim_events --n-cores-max 10 --save-pdf $folder/plot_data/posterior_$i.txt"
	    python3 gen_gals.py --out-directory $folder/sim_events --volume-range $lr >> $foler.log
            python3 est_hubble.py --in-directory $folder/sim_events --n-cores-max 10 --save-pdf $folder/plot_data/$out_name >> $foler.log
	done
done
