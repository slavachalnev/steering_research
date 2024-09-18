import React, { useState } from "react";
import { ProcessedFeaturesType } from "../types";
import {
	BORDER_RADIUS,
	CONTAINER_WIDTH,
	FONT_SIZE,
	PADDING,
	getActivations,
} from "../utils";

const Explorer = ({
	setProcessedFeatures,
}: {
	setProcessedFeatures: React.Dispatch<
		React.SetStateAction<ProcessedFeaturesType[]>
	>;
}) => {
	const [featureInput, setFeatureInput] = useState("");
	const [processingState, setProcessingState] = useState("");

	const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (featureInput.trim() !== "") {
			setProcessingState("Processing features...");

			// Parse input string to array of numbers
			const features = featureInput
				.split(",")
				.map((f) => parseInt(f.trim()))
				.filter((f) => !isNaN(f));

			try {
				const activations = await getActivations(features);
				console.log(activations);
				setProcessingState("");

				setProcessedFeatures(activations);
				// setUniqueFeatures(activations);
				setFeatureInput("");
			} catch (error) {
				console.error("Error processing features:", error);
				setProcessingState("Error processing features");
			}
		}
	};

	return (
		<div style={{ width: CONTAINER_WIDTH }}>
			<form
				onSubmit={handleSubmit}
				style={{ marginTop: "60px", width: "100%" }}
			>
				<textarea
					value={featureInput}
					onChange={(e) => setFeatureInput(e.target.value)}
					placeholder="Enter feature numbers separated by commas (e.g., 1, 2, 3)"
					rows={5}
					style={{
						width: "100%",
						marginBottom: "10px",
						borderRadius: BORDER_RADIUS,
						fontSize: FONT_SIZE,
						padding: PADDING,
					}}
				/>
				<div style={{ display: "flex", flexDirection: "row" }}>
					<button
						type="submit"
						style={{
							backgroundColor: "white",
							width: "fit-content",
							border: "none",
							color: "grey",
							textAlign: "center",
							textDecoration: "none",
							display: "inline-block",
							margin: "4px 2px",
							cursor: "pointer",
							whiteSpace: "nowrap",
							borderRadius: BORDER_RADIUS,
							fontSize: FONT_SIZE,
							padding: PADDING,
						}}
					>
						Process Features
					</button>
					<div style={{ marginLeft: "10px" }}>{processingState}</div>
				</div>
			</form>
		</div>
	);
};

export default Explorer;
