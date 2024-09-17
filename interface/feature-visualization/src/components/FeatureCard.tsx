import "../App.css";

import { useState, useEffect, useRef } from "react";
import { Activation } from "../types";
import { FeatureCardCommands } from "./FeatureCardCommands";
import { TestSamples } from "./TestSamples";
import TokenDisplay from "./TokenDisplay";

const ActivationItem = ({
	activation,
	maxAct,
	lastItem = false,
}: {
	activation: Activation;
	maxAct: number;
	lastItem?: boolean;
}) => {
	// Find the index of the token with the highest value
	const maxValueIndex = activation.values.indexOf(
		Math.max(...activation.values)
	);

	// Calculate the start index, ensuring it's not negative
	const startIndex = Math.max(0, maxValueIndex - 10);

	return (
		<div
			style={{
				position: "relative",
				paddingBottom: "7px",
				paddingTop: "7px",
				textAlign: "left",
				fontSize: ".75rem",
				borderBottom: lastItem ? "none" : "1px solid rgba(0, 0, 0, 0.1)",
				userSelect: "none",
				overflow: "hidden",
				whiteSpace: "nowrap",
			}}
		>
			<div style={{ display: "inline-block" }}>
				{activation.tokens.slice(startIndex).map((token: string, i: number) => (
					<TokenDisplay
						key={i + startIndex}
						index={i + startIndex}
						token={token}
						value={activation.values[i + startIndex]}
						maxValue={maxAct}
					/>
				))}
			</div>
		</div>
	);
};

const SampleToggle = ({
	setExpanded,
	expanded,
}: {
	setExpanded: (id: boolean) => void;
	expanded: boolean;
}) => {
	return (
		<div
			onClick={() => setExpanded(!expanded)}
			style={{
				cursor: "pointer",
				background: "none",
				border: "none",
				fontSize: "16px",
				lineHeight: 1,
				transition: "transform 0.3s ease",
				transform: expanded ? "rotate(-180deg)" : "rotate(0deg)",
				color: "black",
				userSelect: "none",
				textAlign: "center",
			}}
			aria-label={expanded ? "Collapse" : "Expand"}
		>
			▼
		</div>
	);
};

export function FeatureCardSubHeader({ text }: { text: string }) {
	return (
		<div
			style={{
				// fontSize: "12px",
				// fontSize: ".75rem",
				color: "grey",
				fontWeight: "bold",
				marginTop: "4px",
			}}
		>
			{text}
		</div>
	);
}

function FeatureCard({
	feature,
	featureId,
	onDelete,
	onMagnify,
	activations = [],
	maxAct,
}: {
	feature: number;
	featureId: string;
	onDelete: (id: string) => void;
	onMagnify: (id: string) => void;
	activations: Activation[];
	maxAct: number;
}) {
	const [description, setDescription] = useState("");
	const [expanded, setExpanded] = useState(false);
	const [contentHeight, setContentHeight] = useState("0px");
	const contentRef = useRef<HTMLDivElement>(null);
	const [opacity, setOpacity] = useState(0);

	useEffect(() => {
		if (contentRef.current) {
			setContentHeight(
				expanded ? `${contentRef.current.scrollHeight}px` : "0px"
			);
		}
	}, [expanded, activations]);

	useEffect(() => {
		if (expanded) {
			setOpacity(0);
			setTimeout(() => setOpacity(1), 50);
		} else {
			setOpacity(0);
		}
	}, [expanded]);

	return (
		<div
			style={{
				backgroundColor: "rgba(250, 250, 248, 1)",
				border: "6px solid rgba(0, 0, 0, 0)",
				padding: "8px",
				paddingBottom: "4px",
				borderRadius: "15px",
				margin: "0 12px",
				color: "black",
				width: "calc(100% - 24px)",
				maxWidth: "650px",
				position: "relative",
			}}
		>
			{/* CARD HEADER */}
			<div
				style={{
					display: "flex",
					flexDirection: "row",
					color: "black",
				}}
			>
				<FeatureCardCommands
					onDelete={onDelete}
					onMagnify={onMagnify}
					featureId={featureId}
				/>
				<div
					style={{
						borderRadius: "8px",
						backgroundColor: "rgba(42, 97, 211, .7)",
						padding: "4px",
						fontSize: "1rem",
						fontWeight: "bold",

						width: "fit-content",
						height: "fit-content",
						whiteSpace: "nowrap",
						color: "white",
					}}
				>
					{feature}
				</div>
				<div
					style={{
						fontSize: ".75rem",
						marginLeft: "5px",
						marginRight: "45px",
						padding: "1px",
						fontWeight: "bold",
						textAlign: "left",
						color: "rgba(0, 0, 0, 0.5)",
					}}
				>
					{description}
				</div>
			</div>

			<div
				style={{
					padding: "4px",
				}}
			>
				<TestSamples feature={feature} maxAct={maxAct} />
			</div>
			<div
				style={{
					padding: "4px 4px 0px 4px",
				}}
			>
				<FeatureCardSubHeader text="Max activating samples" />

				{activations
					.slice(0, 3)
					.map((activation: Activation, index: number) => (
						<ActivationItem
							key={index}
							activation={activation}
							maxAct={maxAct}
							lastItem={index === 2 && opacity === 0}
						/>
					))}

				<div
					ref={contentRef}
					style={{
						transition: "height 0.3s ease-in-out, opacity 0.5s ease-in-out",
						height: contentHeight,
						overflow: "hidden",
						opacity: opacity,
					}}
				>
					{activations
						.slice(3, 10)
						.map((activation: Activation, index: number) => (
							<ActivationItem
								key={index + 3}
								activation={activation}
								maxAct={maxAct}
								lastItem={index === 6}
							/>
						))}
				</div>

				{activations.length > 3 && (
					<SampleToggle expanded={expanded} setExpanded={setExpanded} />
				)}
			</div>
		</div>
	);
}

export default FeatureCard;
