simulate: sim_events
	python gen_gals.py
	python est_hubble.py --type sim_events

GW_data: GW_Events
	python query.py
	python est_hubble.py --type GW_events

sim_events:
	mkdir -p sim_events

GW_events:
	mkdir -p GW_events

clean:
	rm -r sim_events
	rm -r GW_events
