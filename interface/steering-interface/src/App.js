import React, { useState, useEffect, useRef } from "react";

import "./App.css";
import starterData from "./data.json";

import Results from "./components/Results";
import ControlPanel from "./components/ControlPanel";

function App() {
	const [input, setInput] = useState("I think");
	const [newFeature, setNewFeature] = useState(null);
	const [features, setFeatures] = useState([
		[10138, "0"],
		[2378, "0"],
		[11067, "0"],
	]);
	const [results, setResults] = useState(starterData);

	const [jobs, setJobs] = useState([]);

	useEffect(() => {
		console.log(jobs);
	}, [jobs]);

	useEffect(() => {
		console.log(results);
	}, [results]);

	return (
		<div className="container">
			<ControlPanel setJobs={setJobs} setResults={setResults} />
			{/* <div className="control">
				<AutoResizeTextArea
					value={input}
					onChange={(ev) => setInput(ev.target.value)}
				/>
				{features.length > 0 && (
					<div
						style={{
							fontWeight: "bold",
							marginTop: "1rem",
						}}
					>
						Features:
					</div>
				)}
				{features.map((feature, i) => {
					return (
						<div>
							<div>
								{steeringData[feature[0]]
									? steeringData[feature[0]]
									: feature[0]}
							</div>
							<input
								className="feature-scale"
								type="range"
								min="0"
								max="100"
								value={feature[1]}
								onChange={(ev) => {
									let newScales = [...features];
									newScales[i][1] = ev.target.value;
									setFeatures(newScales);
								}}
							/>
						</div>
					);
				})}

				<button onClick={handleSubmit}>Submit</button>

				<div className="bottom-controls">
					<input
						className="feature-input"
						text="text"
						value={newFeature}
						onChange={(ev) => setNewFeature(ev.target.value)}
						onKeyDown={(ev) => {
							if (ev.code == "Enter") {
								addNewFeature();
							}
						}}
					/>
					<button className="add-feature" onClick={addNewFeature}>
						Add feature
					</button>
				</div>
			</div> */}
			<div className="results">
				{jobs.length > 0 && (
					<div className="process-queue">
						<div className="process-queue-text">Processing {jobs.length}</div>
						<div className="process-queue-animation">
							<div className="dot"></div>
							<div className="dot"></div>
							<div className="dot"></div>
						</div>
					</div>
				)}
				<Results results={results} />
			</div>
		</div>
	);
}

export default App;
