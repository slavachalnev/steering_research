import React, { useState, useEffect } from "react";
import "./App.css"; // Assuming you have a CSS file for styling

function App() {
	const [query, setQuery] = useState("");
	const [results, setResults] = useState([]);

	useEffect(() => {
		const fetchResults = async () => {
			if (query.trim() === "") {
				setResults([]);
				return;
			}
			try {
				const response = await fetch(
					`http://192.168.5.100:5000/search/${query}`
				);
				const data = await response.json();
				setResults(data);
			} catch (error) {
				console.error("Error fetching search results:", error);
			}
		};

		const delayDebounceFn = setTimeout(() => {
			fetchResults();
		}, 300); // Debounce delay

		return () => clearTimeout(delayDebounceFn);
	}, [query]);

	return (
		<div className="search-container">
			<h3>Gemma 2b feature search</h3>
			<input
				type="text"
				value={query}
				onChange={(e) => setQuery(e.target.value)}
				placeholder="Search..."
				className="search-input"
			/>
			{results.length > 0 && (
				<div className="search-results">
					{results.map((result, index) => (
						<div
							key={index}
							className="search-result-item"
							onClick={() => {
								window.open(
									`https://www.neuronpedia.org/gemma-2b/6-res-jb/${result[1]}`,
									"_blank"
								);
							}}
						>
							<span className="result-number">{result[1]}</span>
							<span className="result-description">{result[0]}</span>
						</div>
					))}
				</div>
			)}
		</div>
	);
}

export default App;
