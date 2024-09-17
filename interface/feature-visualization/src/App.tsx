import { useEffect, useState } from "react";
import "./App.css";
import FeatureColumn from "./components/FeatureColumn";
import TokenDisplay from "./components/TokenDisplay";

import { getBaseUrl } from "./utils";
import bret_tokens from "./data/base_bret_tokens_small.json";
import bret_activations from "./data/base_bret_activations_small.json";
import bret_max_activations from "./data/base_bret_max_activations.json";

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

export default function App() {
	const [text, setText] = useState("");
	const [passage, setPassage] = useState(
		"The greatest leaps in the progress of civilization have come from new forms for seeing and discussing ideas, such as written language, printed books, and mathematical notation. Each of these mediums enabled humanity to think and communicate in ways that were previously inconceivable."
	);
	const [features, setFeatures] = useState<number[]>(
		bret_tokens.results.map((data) => data.index)
	);
	const [processedFeatures, setProcessedFeatures] =
		useState<ProcessedFeaturesType[]>(bret_activations);

	const [tokens, setTokens] = useState(bret_tokens.tokens);
	const [activations, setActivations] = useState(bret_tokens.activations);

	const [processingState, setProcessingState] = useState("");
	const [magnified, setMagnified] = useState<number>(-1);

	const [maxActivations, setMaxActivations] = useState<Record<string, number>>(
		{}
	);

	// Max activating
	const [focusToken, setFocusToken] = useState<number>(-1);
	const [processedTokenFeatures, setProcessedTokenFeatures] = useState<
		ProcessedFeaturesType[]
	>([]);

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
	};

	const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (text.trim() != "") {
			processText();

			setProcessedFeatures([]);
			setActivations([]);
			setTokens([]);
			setFeatures([]);
			setPassage("");
		}
	};

	const get_activations = async (features: number[]) => {
		console.log(features);
		setProcessingState("Getting feature visualizations");

		const url = `${getBaseUrl()}/get_feature?features=${features.join(",")}`;
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

		return dataWithId;
	};

	const onMagnify = (id: string) => {
		const index: number = processedFeatures.findIndex(
			(feature) => feature.id === id
		);
		setMagnified(index != -1 ? processedFeatures[index].feature : -1);
	};

	const removeFeature = (id: string) => {
		const indexToRemove = processedFeatures.findIndex(
			(feature) => feature.id == id
		);

		if (indexToRemove !== -1) {
			setProcessedFeatures((prevFeatures) =>
				prevFeatures.filter((_, index) => index !== indexToRemove)
			);
			setActivations((prevActivations) =>
				prevActivations.map((row) =>
					row.filter((_, index) => index !== indexToRemove)
				)
			);

			if (magnified == processedFeatures[indexToRemove].feature) {
				setMagnified(-1);
			}
		}
	};

	const inspectToken = async (index: number) => {
		if (focusToken == index) {
			setFocusToken(-1);
			setProcessedTokenFeatures([]);
		} else {
			setFocusToken(index);
			setMagnified(-1);
			let tokenFeatures = bret_max_activations
				.slice(1)
				[index].top_activations.map((activation) => activation.index);

			const data = await get_activations(tokenFeatures);
			setProcessedTokenFeatures(data);
		}
	};

	const [isInitialMount, setIsInitialMount] = useState(true);

	const handleGetActivations = async () => {
		const data = await get_activations(features);
		setProcessedFeatures([...data]);
		setText("");
	};

	useEffect(() => {
		if (isInitialMount) {
			setIsInitialMount(false);
		} else if (features.length > 0) {
			handleGetActivations();
		}
	}, [features]);

	useEffect(() => {
		const maxActivations = processedFeatures.map((feature) => {
			return {
				[feature.feature]: feature.feature_results.reduce((max, current) =>
					current.maxValue > max.maxValue ? current : max
				).binMax,
			};
		});
		const result = Object.assign({}, ...maxActivations);
		setMaxActivations(result);
	}, [processedFeatures]);

	const containerWidth = "675px";
	const borderRadius = "4px";
	const fontSize = "1rem";
	const padding = "10px";

	return (
		<div
			style={{
				display: "flex",
				flexDirection: "column",
				alignItems: "center",
				width: "100vw",
			}}
		>
			<form
				onSubmit={handleSubmit}
				style={{ marginTop: "60px", width: containerWidth }}
			>
				<textarea
					value={text}
					onChange={(e) => setText(e.target.value)}
					placeholder="Paste your passage here"
					rows={5}
					style={{
						width: "100%",
						marginBottom: "10px",
						fontSize,
						borderRadius,
						padding,
					}}
				/>
				<button
					type="submit"
					style={{
						backgroundColor: "white",
						border: "none",
						color: "grey",
						padding,
						textAlign: "center",
						textDecoration: "none",
						display: "inline-block",
						fontSize,
						margin: "4px 2px",
						cursor: "pointer",
						borderRadius,
						transition: "background-color 0.3s",
					}}
				>
					Process Passage
				</button>
			</form>
			<div style={{ width: containerWidth }}>{processingState}</div>
			<div
				style={{
					position: "relative",
					padding: `${padding} 0`,
					textAlign: "left",
					fontSize: ".75rem",
					borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
					userSelect: "none",
					overflow: "hidden",
					color: "white",
					width: containerWidth,
				}}
			>
				<div style={{ display: "inline-block" }}>
					{tokens.slice(1).map((token: string, i: number) => {
						const magnifiedIndex = processedFeatures.findIndex(
							(feature) => feature.feature === magnified
						);
						const value =
							magnified !== -1 ? activations[i + 1][magnifiedIndex] : 0;
						const maxValue = magnified !== -1 ? maxActivations[magnified] : 0;
						return (
							<TokenDisplay
								key={i}
								index={i}
								token={token}
								value={focusToken != -1 ? (focusToken == i ? 1 : 0) : value}
								maxValue={focusToken != -1 ? 1 : maxValue}
								backgroundColor={focusToken != -1 ? "211, 56,43" : undefined}
								color={"white"}
								fontSize={"18px"}
								inspectToken={inspectToken}
							/>
						);
					})}
				</div>
			</div>
			{focusToken == -1 && processedFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={onMagnify}
					processedFeatures={processedFeatures}
					removeFeature={removeFeature}
				/>
			)}
			{/* {processedTokenFeatures.length > 0 && (
				<FeatureColumn
					onMagnify={onMagnify}
					processedFeatures={processedTokenFeatures}
					removeFeature={removeFeature}
				/>
			)} */}
		</div>
	);
}
