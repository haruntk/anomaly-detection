import axios from "axios";

const API_BASE = "http://localhost:5000";

export const predictImage = async (imageFile, model) => {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model", model);

  try {
    const response = await axios.post(`${API_BASE}/predict`, formData);
    return response.data;
  } catch (err) {
    console.error("API Error:", err);
    return { result: "error", score: 0 };
  }
};
