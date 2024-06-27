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
	]);

	const addNewFeature = async () => {
		const data = await getNeuronpedia(newFeature);
		setFeatures([...features, { id: newFeature, value: 40, color: "red" }]);
		setNewFeature("");
	};

	const handleSubmit = async () => {
		const steering = JSON.stringify(
			features
				.filter((feature) => feature.value > 0)
				.map((feature) => [feature.id, parseInt(feature.value)])
		);

		const data = {
			version:
				"806d4b25f02fbffee8076a34423ecdf8e261774c75adde941e17ed3a49457712",
			input: {
				prompt: input,
				steering: steering,
				n_samples: 4,
				batch_size: 1,
				max_new_tokens: 50,
			},
		};

		const jobId = uuidv4();
		setJobs((prevJobs) => [...prevJobs, jobId]);

		try {
			const response = await predictData(data);
			if (response.status === "succeeded") {
				setResults((prevResults) => [
					...prevResults,
					{
						texts: response.output,
						metadata: data.input,
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
			<FeatureMixer features={features} setFeatures={setFeatures} />

			<button onClick={handleSubmit}>Submit</button>

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
