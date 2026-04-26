const stateUrl = "/api/state";
const stopUrl = "/api/stop";

function statusClass(status) {
  if (status === "confirmed") return "confirmed";
  if (status === "candidate") return "candidate";
  return "unknown";
}

function titleForEvent(event) {
  if (event.status === "confirmed") return event.name || "Confirmed person";
  if (event.status === "candidate") return event.name || "Candidate match";
  return "Unknown person";
}

function renderAlerts(events) {
  const container = document.getElementById("alerts");
  const unknownEvents = events.filter((event) => event.status === "unknown");
  container.innerHTML = "";

  if (!unknownEvents.length) {
    container.innerHTML = '<div class="empty">No unknown alerts</div>';
    return;
  }

  unknownEvents.slice(0, 6).forEach((event) => {
    const item = document.createElement("div");
    item.className = `alert-card ${statusClass(event.status)}`;
    const snapshot = event.snapshot_url
      ? `<img src="${event.snapshot_url}" alt="${event.status || "event"} face snapshot" />`
      : '<div class="snapshot-placeholder">No image</div>';

    item.innerHTML = `
      ${snapshot}
      <div class="alert-content">
        <strong>${titleForEvent(event)}</strong>
        <span class="pill ${statusClass(event.status)}">${event.status || ""}</span>
        <div class="alert-meta">
          <span>#${event.track_id || ""}</span>
          <span>score ${Number(event.avg_score || 0).toFixed(2)}</span>
          <span>votes ${event.votes || 0}</span>
        </div>
        <small>${event.timestamp || ""}</small>
      </div>
    `;
    container.appendChild(item);
  });
}

function renderEvents(events) {
  const tbody = document.getElementById("events");
  tbody.innerHTML = "";

  if (!events.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty">No events yet</td></tr>';
    return;
  }

  events.forEach((event) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${event.timestamp || ""}</td>
      <td>#${event.track_id || ""}</td>
      <td>${event.name || ""}</td>
      <td><span class="pill ${statusClass(event.status)}">${event.status || ""}</span></td>
      <td>${event.avg_score || ""}</td>
      <td>${event.votes || ""}</td>
      <td>${event.frame_number || ""}</td>
      <td>${event.snapshot_url ? `<a href="${event.snapshot_url}" target="_blank">view</a>` : ""}</td>
    `;
    tbody.appendChild(row);
  });
}

async function refreshState() {
  try {
    const response = await fetch(stateUrl, { cache: "no-store" });
    const state = await response.json();

    if (!state.ready) return;

    const metrics = state.metrics;
    document.getElementById("source").textContent = metrics.video_source || "Video source unavailable";
    document.getElementById("runtime").textContent = `Runtime: ${metrics.runtime_mode || "--"}`;
    document.getElementById("running").textContent = metrics.running ? "Running" : "Stopped";
    document.getElementById("running").className = metrics.running ? "ok" : "stopped";
    document.getElementById("fps").textContent = Number(metrics.average_fps || 0).toFixed(1);
    document.getElementById("activeTracks").textContent = metrics.active_tracks || 0;
    document.getElementById("registeredPeople").textContent = metrics.registered_people || 0;
    document.getElementById("confirmed").textContent = metrics.confirmed_recognitions || 0;

    renderEvents(state.events || []);
    renderAlerts(state.events || []);
  } catch (error) {
    document.getElementById("running").textContent = "Disconnected";
    document.getElementById("running").className = "stopped";
  }
}

refreshState();
setInterval(refreshState, 1000);

document.getElementById("stopButton").addEventListener("click", async () => {
  const button = document.getElementById("stopButton");
  button.disabled = true;
  button.textContent = "Stopping...";

  try {
    await fetch(stopUrl, { method: "POST" });
    await refreshState();
    button.textContent = "Stopped";
  } catch (error) {
    button.disabled = false;
    button.textContent = "Stop Camera";
  }
});
