import React, { useState } from "react";
import { predictImage } from "../services/api";

function ImageUploader() {
  const [image, setImage] = useState(null);
  const [model, setModel] = useState("padim");
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setImage(e.target.files[0]);
    setResult(null); // eski sonucu temizle
  };

  const handleSubmit = async () => {
    if (!image) return alert("Please select an image");
    const res = await predictImage(image, model);
    setResult(res);
  };

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      
      <select value={model} onChange={(e) => setModel(e.target.value)}>
        <option value="padim">PaDiM</option>
        <option value="spade">SPADE (yakında)</option>
        <option value="patchcore">PatchCore (yakında)</option>
      </select>

      <button onClick={handleSubmit}>Detect</button>

      {result && (
        <div style={{ marginTop: "1rem" }}>
          <h4>Result: {result.result}</h4>
          <p>Score: {result.score.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
