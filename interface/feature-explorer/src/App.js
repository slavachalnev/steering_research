import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import FeatureView from "./FeatureView";
import { fetchData } from "./utils";

function App() {
	const iframeRef = useRef();
	const [value, setValue] = useState(0);
	const [history, setHistory] = useState([]);
	const [featureNumber, setFeatureNumber] = useState(0);
	const [features, setFeatures] = useState([10138]);
	const [featureDescription, setFeatureDescription] = useState("");
	const [similar, setSimilar] = useState(null);

	const historyRef = useRef(null);
	const [showScrollButton, setShowScrollButton] = useState(false);

	useEffect(() => {
		const handleScroll = () => {
			if (historyRef.current) {
				setShowScrollButton(historyRef.current.scrollLeft > 0);
			}
		};

		const historyElement = historyRef.current;
		if (historyElement) {
			historyElement.addEventListener("scroll", handleScroll);
		}

		return () => {
			if (historyElement) {
				historyElement.removeEventListener("scroll", handleScroll);
			}
		};
	}, []);

	const scrollToStart = () => {
		if (historyRef.current) {
			historyRef.current.scrollTo({ left: 0, behavior: "smooth" });
		}
	};

	const fetchDescription = async (featureNumber) => {
		try {
			const response = await fetch(
				`http://localhost:5000/get_description?key=${featureNumber}`
			);
			if (!response.ok) {
				throw new Error("Failed to fetch description");
			}
			const data = await response.json();
			setFeatureDescription(data.description);
			setHistory([
				...history.filter((item) => item.feature !== featureNumber),
				{ feature: featureNumber, description: data.description },
			]);
		} catch (error) {
			console.error("Error fetching description:", error);
			setFeatureDescription("");
			setHistory([
				...history.filter((item) => item.feature !== featureNumber),
				{ feature: featureNumber, description: "" },
			]);
		}
		scrollToStart();
	};

	const getData = async (feature) => {
		const data = await fetchData(feature);
		setSimilar(data);
	};

	useEffect(() => {
		fetchDescription(featureNumber);
		getData(featureNumber);
	}, [featureNumber]);

	useEffect(() => {
		console.log(features);
	}, [features]);

	return (
		<div className="App">
			<div className="neuron-history-container">
				<div className="neuron-history row" ref={historyRef}>
					{[...history].reverse().map((item) => {
						return (
							<div onClick={(ev) => setFeatureNumber(item.feature)}>
								<div>
									<b>{item.feature}</b>
								</div>
								<div>{item.description.slice(0, 30) + "..."}</div>
							</div>
						);
					})}
					{showScrollButton && (
						<button
							className="scroll-left-button"
							onClick={scrollToStart}
						></button>
					)}
				</div>
			</div>
			<div className="neuron-view row">
				<div className="neuron-bar column">
					<h3>Find by</h3>
					<div className="input-group">
						<input
							type="number"
							value={value}
							onChange={(e) => setValue(e.target.value)}
							placeholder="Enter feature number"
						/>
						<button onClick={() => setFeatureNumber(value)}>Set</button>
					</div>
					<div>
						<input
							style={{
								width: "100%",
							}}
							type="text"
							placeholder="Search by description"
						/>
					</div>
				</div>
				<div className="neuron-bar row">
					<div className="columm">
						<div className="row">
							<h3>Feature {featureNumber}</h3>
							<button
								onClick={() => setFeatures((prev) => [...prev, featureNumber])}
							>
								Add to Features
							</button>
						</div>
						<p>{featureDescription}</p>
						<div
							style={{
								color: "gray",
							}}
						>
							Related features
						</div>
						<div className="row wrap">
							{similar &&
								similar.indices.slice(1).map((index) => {
									return (
										<div
											onClick={(ev) => setFeatureNumber(index)}
											className="related-neurons"
										>
											{index}
										</div>
									);
								})}
						</div>
					</div>
				</div>
				<iframe
					ref={iframeRef}
					src={
						"https://neuronpedia.org/gemma-2b/6-res-jb/" +
						featureNumber +
						"?embed=true"
					}
					title="Neuronpedia"
					className="neuronpedia-iframe"
				></iframe>
			</div>
			{/* <div className="iframe-container">
				<h2>Feature {featureNumber}</h2>
				<iframe
					ref={iframeRef}
					src={
						"https://neuronpedia.org/gemma-2b/6-res-jb/" +
						featureNumber +
						"?embed=true"
					}
					title="Neuronpedia"
					className="neuronpedia-iframe"
				></iframe>
			</div> */}
			{[...features].reverse().map((feature) => {
				console.log(feature);
				return (
					<FeatureView feature={feature} setFeatureNumber={setFeatureNumber} />
				);
			})}
		</div>
	);
}

export default App;
