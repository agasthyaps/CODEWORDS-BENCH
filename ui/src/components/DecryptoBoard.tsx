type Props = {
  redKey: string[];
  blueKey: string[];
};

export default function DecryptoBoard({ redKey, blueKey }: Props) {
  return (
    <div className="decrypto-keys">
      <div className="key-row red">
        <span className="key-label">RED KEY</span>
        {redKey.map((word, idx) => (
          <div key={`red-${word}-${idx}`} className="key-card">
            {idx + 1}. {word}
          </div>
        ))}
      </div>
      <div className="key-row blue">
        <span className="key-label">BLUE KEY</span>
        {blueKey.map((word, idx) => (
          <div key={`blue-${word}-${idx}`} className="key-card">
            {idx + 1}. {word}
          </div>
        ))}
      </div>
    </div>
  );
}
