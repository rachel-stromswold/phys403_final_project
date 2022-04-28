simulate_plot_CI: generate_events plot_data
	python3 est_hubble.py --type sim_events --save-intervals plot_data/intervals.txt --n-cores-max 10
	python3 plot_intervals.py --interval-file plot_data intervals.txt --plot-fname plot_data/intervals.pdf

simulate: generate_events
	python3 est_hubble.py --type sim_events

generate_events: sim_events
	rm -f sim_events/*
	python3 gen_gals.py

plot_data:
	mkdir -p plot_data

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
