import { useState } from "react";
import { Card, Form, Button, Alert } from "react-bootstrap";
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [city, setCity] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("male");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult({ loading: true });

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ city, age: parseInt(age), gender }),
});

      if (!response.ok) throw new Error("Error fetching prediction");

      const data = await response.json();
      setResult({ loading: false, data });
    } catch (err) {
      setResult({ loading: false, error: err.message });
    }
  };

  return (
    <div className="App" style={{ minHeight: "100vh", background: "#f0f2f5", paddingTop: "50px" }}>
      <Card style={{ maxWidth: "400px", margin: "auto", padding: "20px", borderRadius: "10px", boxShadow: "0 4px 10px rgba(0,0,0,0.1)" }}>
        <Card.Title className="text-center mb-4">Health Risk Prediction</Card.Title>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Control
              type="text"
              placeholder="City"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              required
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Control
              type="number"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              required
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Select value={gender} onChange={(e) => setGender(e.target.value)}>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </Form.Select>
          </Form.Group>

          <Button variant="primary" type="submit" className="w-100">Predict</Button>
        </Form>

        <div style={{ marginTop: "20px" }}>
          {result?.loading && <Alert variant="info">Fetching prediction...</Alert>}
          {result?.error && <Alert variant="danger">{result.error}</Alert>}
          {result?.data && (
            <Card body className="mt-3">
              <p><b>HealthImpactScore:</b> {result.data.health_score}</p>
              <p><b>Age/Gender Score:</b> {result.data.age_gender_score}</p>
              <p><b>Final Score:</b> {result.data.final_score}</p>
              <p><b>Risk Label:</b> <span style={{ color: result.data.final_score >= 70 ? "red" : result.data.final_score >= 50 ? "orange" : "green" }}>{result.data.risk_label}</span></p>
            </Card>
          )}
        </div>
      </Card>
    </div>
  );
}

export default App;
