import React, { useState, useEffect } from "react";

import "./App.css";
import starterData from "./data.json";

import Results from "./components/Results";
import ControlPanel from "./components/ControlPanel";

function App() {
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
