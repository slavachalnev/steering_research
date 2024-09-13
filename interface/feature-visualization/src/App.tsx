import { useEffect, useState } from "react";
import "./App.css";
import FeatureColumn from "./FeatureColumn";
import { TokenDisplay } from "./FeatureCard";

import { getBaseUrl } from "./utils";
import bret_tokens from "./base_bret_tokens_small.json";
import bret_activations from "./base_bret_activations_small.json";

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
	const [text, setText] = useState("");
	const [passage, setPassage] = useState(
		"The greatest leaps in the progress of civilization have come from new forms for seeing and discussing ideas, such as written language, printed books, and mathematical notation. Each of these mediums enabled humanity to think and communicate in ways that were previously inconceivable."
	);
	const [features, setFeatures] = useState<Features[]>(bret_tokens.results);
	const [processedFeatures, setProcessedFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);

	const [tokens, setTokens] = useState(bret_tokens.tokens);
	const [activations, setActivations] = useState(bret_tokens.activations);

	const [processingState, setProcessingState] = useState("");
	const [magnified, setMagnified] = useState<string>("");

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
		processText();
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

	const onMagnify = (id: string) => {
		const index = processedFeatures.findIndex((feature) => feature.id === id);
		setMagnified(index !== -1 ? index.toString() : "");
	};

	useEffect(() => {
		// if (features.length > 0) get_activations();
	}, [features]);

	useEffect(() => {
		// processText();
		// console.log(processedFeatures);
		// console.log(tokens);
		// console.log(activations);
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
			<div
				style={{
					position: "relative",
					paddingBottom: "7px",
					paddingTop: "7px",
					textAlign: "left",
					fontSize: ".75rem",
					borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
					userSelect: "none",
					overflow: "hidden",
					color: "white",
					// whiteSpace: "nowrap",
					maxWidth: "675px",
					marginLeft: "15px",
				}}
			>
				<div style={{ display: "inline-block" }}>
					{tokens.slice(1).map((token: string, i: number) => {
						if (magnified != "") {
							const value = activations[i + 1][parseInt(magnified) as number];
							const maxValue = Math.max(...activations[i + 1]);
							return (
								<TokenDisplay
									key={i}
									token={token}
									value={value}
									maxValue={maxValue}
									color={"white"}
								/>
							);
						}
						return (
							<TokenDisplay
								key={i}
								token={token}
								value={0}
								maxValue={0}
								color={"white"}
							/>
						);
					})}
				</div>
			</div>
			{processedFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={onMagnify}
					processedFeatures={processedFeatures}
					setProcessedFeatures={setProcessedFeatures}
				/>
			)}
		</div>
	);
}
