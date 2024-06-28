import React, { useState } from "react";
import { AutoResizeTextArea } from "./TextArea";
import { v4 as uuidv4 } from "uuid";
import { predictData, getNeuronpedia, steeringData } from "../utils/api";
import FeatureMixer from "./FeatureMixer";

const ControlPanel = ({ setJobs, setResults }) => {
	const [input, setInput] = useState("I think");
	const [newFeature, setNewFeature] = useState(null);
	const [features, setFeatures] = useState([
		{ id: 10138, value: 0, color: "orange" },
		{ id: 2378, value: 0, color: "purple" },
		{ id: 11067, value: 0, color: "blue" },
		{ id: 10200, value: 0, color: "green" },
	]);
	const [nSamples, setNSamples] = useState(4); // New state for n_samples

	const addNewFeature = async () => {
		const data = await getNeuronpedia(newFeature);
		console.log(data);
		// setFeatures([...features, { id: newFeature, value: 40, color: "red" }]);
		setNewFeature("");
	};

	const handleSubmit = async () => {
		const steering = JSON.stringify(
			features
				.filter((feature) => feature.value > 0)
				.map((feature) => [feature.id, parseInt(feature.value)])
		);

		// const systemPrompt =
		// 	"Continue the text below using basic markdown text syntax.\n\n";

		const systemPrompt = "";

		const data = {
			version:
				"806d4b25f02fbffee8076a34423ecdf8e261774c75adde941e17ed3a49457712",
			input: {
				prompt: systemPrompt + input,
				steering: steering,
				n_samples: nSamples, // Use the selected n_samples value
				batch_size: 1,
				max_new_tokens: 50,
			},
		};

		const steeringFeatures = features;

		const jobId = uuidv4();
		setJobs((prevJobs) => [...prevJobs, jobId]);

		try {
			const response = await predictData(data);
			if (response.status === "succeeded") {
				setResults((prevResults) => [
					...prevResults,
					{
						texts: response.output.map((text) =>
							text.slice(systemPrompt.length)
						),
						metadata: Object.assign(data.input, { prompt: input }),
						features: steeringFeatures,
					},
				]);
			}
		} catch (error) {
			console.error("Error in handleSubmit:", error);
		} finally {
			setJobs((prevJobs) => prevJobs.filter((job) => job !== jobId));
		}
	};

	return (
		<div className="control">
			<AutoResizeTextArea
				value={input}
				onChange={(ev) => setInput(ev.target.value)}
			/>

			<br />
			<span className="control-title">Feature direction</span>
			<FeatureMixer features={features} setFeatures={setFeatures} />

			{/* New selection input for n_samples */}
			<label htmlFor="nSamples">Number of Samples:</label>
			<select
				id="nSamples"
				value={nSamples}
				onChange={(ev) => setNSamples(parseInt(ev.target.value))}
			>
				{[4, 5, 6, 7, 8, 9, 10].map((option) => (
					<option key={option} value={option}>
						{option}
					</option>
				))}
			</select>

			<button className="submit-button" onClick={handleSubmit}>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					viewBox="0 0 24 24"
					width="24"
					height="24"
					fill="currentColor"
					aria-label="send"
				>
					<path d="M2.01 21L23 12 2.01 3v7l15 2-15 2z" />
				</svg>
			</button>

			<div className="bottom-controls">
				<input
					className="feature-input"
					text="text"
					value={newFeature}
					onChange={(ev) => setNewFeature(ev.target.value)}
					onKeyDown={(ev) => {
						if (ev.code === "Enter") {
							addNewFeature();
						}
					}}
				/>
				<button className="add-feature" onClick={addNewFeature}>
					Add feature
				</button>
			</div>
		</div>
	);
};

export default ControlPanel;
