"""Live SUMO broadcast server with selectable controllers and phase panel.

Replays nothing: the simulation is stepped in real time by libsumo while a
chosen controller (traditional agent, DQN-family snapshot, or a human
operator) picks signal phases at each decision interval.  The browser page
receives the stream over SSE and sends control commands back over HTTP POST.
"""
