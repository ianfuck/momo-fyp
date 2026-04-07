import type { StatusSnapshot } from "../lib/types";

export function SerialMonitor({ status }: { status: StatusSnapshot }) {
  const monitor = status.serial_monitor;

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Serial Monitor</h3>
        <span>{status.serial_connected ? "Connected" : "Disconnected"}</span>
      </div>
      <div className="serial-overview">
        <SerialRow label="Port" value={monitor.port ?? "-"} />
        <SerialRow label="Baud" value={monitor.baud_rate ? String(monitor.baud_rate) : "-"} />
        <SerialRow label="Last TX" value={monitor.last_tx_at ? `${monitor.last_tx_at}` : "-"} />
        <SerialRow label="Last RX" value={monitor.last_rx_at ? `${monitor.last_rx_at}` : "-"} />
      </div>
      {monitor.last_tx ? (
        <div className="serial-payload">
          <strong>Last Sent</strong>
          <code>{monitor.last_tx}</code>
        </div>
      ) : null}
      {monitor.last_rx ? (
        <div className="serial-payload">
          <strong>Last Received</strong>
          <code>{monitor.last_rx}</code>
        </div>
      ) : null}
      {monitor.last_error ? (
        <div className="update-feedback error">
          <strong>Last Serial Error</strong>
          <div>{monitor.last_error}</div>
        </div>
      ) : null}
      <div className="serial-log">
        {monitor.entries.length === 0 ? (
          <p>No serial traffic yet.</p>
        ) : (
          monitor.entries.map((entry, index) => (
            <div key={`${entry.ts}-${entry.direction}-${index}`} className={`serial-entry ${entry.direction}`}>
              <span>{entry.ts}</span>
              <strong>{entry.direction.toUpperCase()}</strong>
              <code>{entry.message}</code>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function SerialRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="runtime-item">
      <strong>{label}</strong>
      <span>{value}</span>
    </div>
  );
}
