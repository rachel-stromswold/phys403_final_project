simulate: sim_events
	python3 gen_gals.py
	python3 est_hubble.py --type sim_events

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
