import { useEffect, useState } from "react";
import "./App.css";
import FeatureColumn from "./FeatureColumn";
import { getBaseUrl } from "./utils";
import bret_tokens from "./base_bret_tokens.json";
import bret_activations from "./base_bret_activations.json";

interface FeatureData {
	binMax: number;
	binMin: number;
	maxValue: number;
	minValue: number;
	tokens: string[];
	values: number[];
}

interface ProcessedFeaturesType {
	feature: number;
	id: string;
	feature_results: FeatureData[];
}

interface Features {
	difference: number;
	index: number;
}

export default function App() {
	const [text, setText] = useState(
		"The greatest leaps in the progress of civilization have come from new forms for seeing and discussing ideas, such as written language, printed books, and mathematical notation. Each of these mediums enabled humanity to think and communicate in ways that were previously inconceivable."
	);
	const [passage, setPassage] = useState("");
	const [features, setFeatures] = useState<Features[]>(bret_tokens.results);
	const [processedFeatures, setProcessedFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);

	const [tokens, setTokens] = useState([bret_tokens.tokens]);
	const [activations, setActivations] = useState([bret_tokens.activations]);

	const [processingState, setProcessingState] = useState("");

	const processText = async () => {
		setPassage(text);
		setProcessingState("Getting unique activations for text");
		console.log("Submitted passage:", text);
		const url = `${getBaseUrl()}/get_unique_acts?text=${text}`;
		const response = await fetch(url, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			},
		});
		const data = await response.json();
		console.log(data);
		setFeatures(data.results);
		setActivations(data.activations);
		setTokens(data.tokens);
		setText("");
	};

	const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		// TODO: Implement passage processing logic
		// processText();
	};

	const get_activations = async () => {
		console.log(features);
		setProcessingState("Getting feature visualizations");

		const url = `${getBaseUrl()}/get_feature?features=${features
			.map((d) => d.index)
			.join(",")}`;
		const response = await fetch(url, {
			method: "GET",
			headers: {
				"Content-Type": "application/json",
			},
		});
		const data = await response.json();
		const dataWithId: ProcessedFeaturesType[] = data.map((d: any) => {
			return Object.assign(d, {
				id: crypto.randomUUID(),
			});
		});

		console.log(dataWithId);
		setProcessingState("");
		setProcessedFeatures([...dataWithId]);
	};

	useEffect(() => {
		// if (features.length > 0) get_activations();
	}, [features]);

	useEffect(() => {
		console.log(bret_tokens);
		console.log(bret_activations);
	}, []);

	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				// justifyContent: "space-between",
				// transform: inspectedFeature
				// 	? "translateX(0)"
				// 	: "translateX(calc(50% - 50%))", // Center the content
				width: "100vw",
			}}
		>
			<form
				onSubmit={handleSubmit}
				style={{ margin: "auto", marginTop: "60px", width: "800px" }}
			>
				<textarea
					value={text}
					onChange={(e) => setText(e.target.value)}
					placeholder="Paste your passage here"
					rows={5}
					style={{ width: "800px", marginBottom: "10px" }}
				/>
				<button type="submit">Process Passage</button>
			</form>
			<div>{processingState}</div>
			<p>{passage}</p>
			{processedFeatures.length > 0 && (
				<FeatureColumn
					processedFeatures={processedFeatures}
					setProcessedFeatures={setProcessedFeatures}
				/>
			)}
		</div>
	);
}
