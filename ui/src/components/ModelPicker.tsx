import { ModelInfo, TeamRoleConfig } from "../types";

type Props = {
  models: ModelInfo[];
  value: TeamRoleConfig;
  onChange: (next: TeamRoleConfig) => void;
  label: string;
};

export default function ModelPicker({ models, value, onChange, label }: Props) {
  const options = models.map((m) => (
    <option key={m.model_id} value={m.model_id}>
      {m.name} ({m.model_id})
    </option>
  ));

  function update(field: keyof TeamRoleConfig, val: string) {
    onChange({ ...value, [field]: val });
  }

  return (
    <div className="panel">
      <h3>{label}</h3>
      <div className="form-row">
        <label>Cluer</label>
        <select value={value.cluer} onChange={(e) => update("cluer", e.target.value)}>
          {options}
        </select>
      </div>
      <div className="form-row">
        <label>Guesser 1</label>
        <select
          value={value.guesser_1}
          onChange={(e) => update("guesser_1", e.target.value)}
        >
          {options}
        </select>
      </div>
      <div className="form-row">
        <label>Guesser 2</label>
        <select
          value={value.guesser_2 || value.guesser_1}
          onChange={(e) => update("guesser_2", e.target.value)}
        >
          {options}
        </select>
      </div>
    </div>
  );
}
