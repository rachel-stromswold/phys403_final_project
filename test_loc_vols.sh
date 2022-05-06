#loc_ranges=("1e6 1e8" "1e6 1e9")
modes=( "uniform" "clusters" )

folder=$(timedatectl | grep Local | awk '{print "runs_"$4 "_"$5}' | sed 's/-/_/g' | sed 's/:/_/g')
mkdir $folder
mkdir $folder/sim_events
mkdir $folder/plot_data
mkdir $folder/data_products
#folder=$(timedatectl | grep Local | awk '{print "runs_"$4 "_"$5}')
timedatectl > "$folder/out.log"

for md in "${modes[@]}"; do
    plt_cmd="--output-prefix $folder/data_products/p_${md}_ --posterior-files"
    for i in {1..5}; do
	    echo "i=$i:"
            #generate clusters
            echo "\tpython3 gen_gals.py --out-directory $folder/sim_events --mode $md"
	    python3 gen_gals.py --out-directory $folder/sim_events --mode $md >> "$folder/out.log"
            #estimate H_0
            echo "\tpython3 est_hubble.py --in-directory $folder/sim_events --n-cores-max 10 --save-pdf $folder/plot_data/posterior_${md}_$i.txt"
            python3 est_hubble.py --in-directory $folder/sim_events --n-cores-max 10 --save-pdf $folder/plot_data/posterior_${md}_$i.txt >> "$folder/out.log"
            #add information to the plotting command
            plt_cmd="$plt_cmd $folder/plot_data/posterior_${md}_$i.txt"
	done
        echo "python gen_plots.py $plt_cmd"
        python gen_plots.py $plt_cmd
done
