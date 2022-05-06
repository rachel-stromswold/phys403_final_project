simulate_plot: generate_events plot_data
	python3 est_hubble.py --in-directory sim_events --n-cores-max 10 --save-pdf plot_data/posterior.txt
	python3 gen_plots.py --output-prefix data_products/ --posterior-files plot_data/posterior.txt

simulate: generate_events
	python3 est_hubble.py --type sim_events

generate_events: sim_events
	rm -f sim_events/*
	python3 gen_gals.py

plot_data:
	mkdir -p plot_data
	mkdir -p data_products

GW_data: GW_Events
	python3 query.py
	python3 est_hubble.py --type GW_events

sim_events:
	mkdir -p sim_events

GW_events:
	mkdir -p GW_events

clean:
	rm -r sim_events
	rm -r GW_events
