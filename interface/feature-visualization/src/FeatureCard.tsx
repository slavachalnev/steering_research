// import { useState, useEffect, useRef } from "react";
import "./App.css";

import { useState, useEffect, useRef, useMemo } from "react";
import { getBaseUrl } from "./utils";

export const TokenDisplay = ({
	index,
	token,
	value,
	maxValue,
	color = "black",
	backgroundColor = "42, 97, 211",
	fontSize = "12px",
	inspectToken = (id: number) => {},
}: {
	index: number;
	token: string;
	value: number;
	maxValue: number;
	color?: string;
	backgroundColor?: string;
	fontSize?: string;
	inspectToken?: (id: number) => void;
}) => {
	const [isHovering, setIsHovering] = useState(false);
	const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
	const spanRef = useRef<HTMLSpanElement>(null);

	// Use useMemo to recalculate opacity when value or maxValue changes
	const opacity = useMemo(() => {
		if (value == 0 && maxValue == 0) return 0;
		return Math.min(0.85, value / maxValue);
	}, [value, maxValue]);

	const displayToken = useMemo(
		() =>
			token.startsWith("▁") || token.startsWith(" ") ? token.slice(1) : token,
		[token]
	);

	const addSpace = useMemo(
		() => (token.includes("▁") || token.startsWith(" ") ? " " : ""),
		[token]
	);

	const updateTooltipPosition = () => {
		if (spanRef.current) {
			const rect = spanRef.current.getBoundingClientRect();
			setTooltipPosition({
				top: rect.top - 24, // 24px above the span
				left: rect.left + rect.width / 2,
			});
		}
	};

	// Update tooltip position when value changes
	useEffect(() => {
		if (isHovering) {
			updateTooltipPosition();
		}
	}, [value, isHovering]);

	return (
		<span
			ref={spanRef}
			style={{
				position: "relative",
				paddingLeft: addSpace ? "4px" : "0px",
				display: "inline-block",
			}}
			onMouseEnter={() => {
				setIsHovering(true);
				updateTooltipPosition();
			}}
			onMouseLeave={() => setIsHovering(false)}
			onClick={() => {
				inspectToken(index);
			}}
		>
			{" "}
			{isHovering && (
				<div
					style={{
						position: "fixed",
						top: `${tooltipPosition.top}px`,
						left: `${tooltipPosition.left}px`,
						transform: "translateX(-50%)",
						backgroundColor: "rgba(0, 0, 0, 0.8)",
						color: "white",
						borderRadius: "4px",
						fontSize: "12px",
						whiteSpace: "nowrap",
						padding: "2px 4px",
						zIndex: 1000,
						pointerEvents: "none",
					}}
				>
					{value.toFixed(2)}
				</div>
			)}
			<span
				style={{
					backgroundColor: `rgba(${backgroundColor}, ${opacity})`,
					display: "inline-block",
					borderRadius: "4px",
					fontSize,
					color,
				}}
			>
				{addSpace}
				{displayToken}
			</span>
		</span>
	);
};

const ActivationItem = ({
	activation,
	maxAct,
}: {
	activation: any;
	maxAct: number;
}) => {
	console.log(activation);

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
				borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
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

const FeatureCardCommands = ({
	onDelete,
	onMagnify,
	onTest,
	featureId,
}: {
	onDelete: (id: string) => void;
	onMagnify: (id: string) => void;
	onTest: () => void;
	featureId: string;
}) => {
	return (
		<div
			style={{
				position: "absolute",
				top: "7px",
				right: "10px",
				display: "flex",
				gap: "8px",
			}}
		>
			<div
				style={{
					cursor: "pointer",
					fontSize: "16px",
					color: "gray",
					transition: "color 0.1s ease-in-out",
				}}
				onClick={() => onTest()}
				onMouseEnter={(e) => (e.currentTarget.style.color = "black")}
				onMouseLeave={(e) => (e.currentTarget.style.color = "gray")}
			>
				<svg
					width="16"
					height="16"
					viewBox="0 0 16 16"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						d="M12.5 1.5L14.5 3.5M9.5 4.5L11.5 2.5L13.5 4.5L11.5 6.5L9.5 4.5ZM2.5 13.5L6.5 9.5L8.5 11.5L4.5 15.5H2.5V13.5Z"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
				</svg>
			</div>
			<div
				style={{
					cursor: "pointer",
					fontSize: "16px",
					color: "gray",
					transition: "color 0.1s ease-in-out",
				}}
				onClick={() => onMagnify(featureId)}
				onMouseEnter={(e) => {
					// onMagnify(featureId);
					e.currentTarget.style.color = "black";
				}}
				onMouseLeave={(e) => {
					// onMagnify("");
					e.currentTarget.style.color = "gray";
				}}
			>
				<svg
					width="16"
					height="16"
					viewBox="0 0 16 16"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						d="M7 11.5C9.48528 11.5 11.5 9.48528 11.5 7C11.5 4.51472 9.48528 2.5 7 2.5C4.51472 2.5 2.5 4.51472 2.5 7C2.5 9.48528 4.51472 11.5 7 11.5Z"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
					<path
						d="M10.5 10.5L13.5 13.5"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
				</svg>
			</div>
			<div
				style={{
					cursor: "pointer",
					fontSize: "16px",
					color: "gray",
					transition: "color 0.1s ease-in-out",
				}}
				onClick={() => onDelete(featureId)}
				onMouseEnter={(e) => (e.currentTarget.style.color = "black")}
				onMouseLeave={(e) => (e.currentTarget.style.color = "gray")}
			>
				<svg
					width="16"
					height="16"
					viewBox="0 0 16 16"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						d="M2 4h12M5.333 4V2.667a1.333 1.333 0 011.334-1.334h2.666a1.333 1.333 0 011.334 1.334V4m2 0v9.333a1.333 1.333 0 01-1.334 1.334H4.667a1.333 1.333 0 01-1.334-1.334V4h9.334z"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
				</svg>
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
			}}
			aria-label={expanded ? "Collapse" : "Expand"}
		>
			▼
		</div>
	);
};

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
	activations: any;
	maxAct: number;
}) {
	const [description, setDescription] = useState("");
	const [expanded, setExpanded] = useState(false);
	const [contentHeight, setContentHeight] = useState("0px");
	const contentRef = useRef<HTMLDivElement>(null);
	const [opacity, setOpacity] = useState(0);

	// Variables related to testing
	const [testText, setTestText] = useState("");
	const [showTest, setShowTest] = useState<boolean>(false);
	const [loading, setLoading] = useState<boolean>(false);
	const testTextRef = useRef<HTMLDivElement>(null);
	const [testActivations, setTestActivations] = useState<any>([
		[
			[0],
			[0],
			[2.229448080062866],
			[0.4582373797893524],
			[0.826278805732727],
			[1.7092781066894531],
			[1.4163553714752197],
		],
	]);
	const [textTokens, setTextTokens] = useState<any>([
		"These",
		"▁are",
		"▁ancient",
		"▁sum",
		"er",
		"ian",
		"▁texts",
	]);

	const handleInput = (e: React.FormEvent<HTMLDivElement>) => {
		const newText = e.currentTarget.textContent || "";
		if (newText !== testText) {
			setTestText(newText);
		}
	};

	const onTest = () => {
		setShowTest(true);
		setTimeout(() => {
			testTextRef.current?.focus();
		}, 100);
	};

	const submitTest = async (e: React.FormEvent<HTMLDivElement>) => {
		setLoading(true);

		try {
			const url = `${getBaseUrl()}/get_max_feature_acts?text=${encodeURIComponent(
				testText
			)}&features=${feature}`;
			const response = await fetch(url, {
				method: "GET",
				headers: {
					"Content-Type": "application/json",
				},
			});

			if (!response.ok) {
				throw new Error("Network response was not ok");
			}

			const data = await response.json();
			console.log("Max feature acts data:", data);
			// Process the data as needed
			setTestActivations(data.activations);
			setTextTokens(data.tokens);
		} catch (error) {
			console.error("Error fetching max feature acts:", error);
			// Handle the error (e.g., show an error message to the user)
		} finally {
			setLoading(false);
			setShowTest(false);
		}

		// setTimeout(() => {
		// 	setLoading(false);
		// 	setShowTest(false);
		// 	e.target.blur();
		// }, 1000);
	};

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
				textAlign: "center",
				width: "calc(100% - 24px)",
				maxWidth: "650px",
				position: "relative",
			}}
		>
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
					onTest={onTest}
					featureId={featureId}
				/>
				<div
					style={{
						borderRadius: "5px",
						backgroundColor: "rgba(42, 97, 211, .7)",
						padding: "1px",
						fontSize: ".75rem",
						width: "fit-content",
						height: "fit-content",
						whiteSpace: "nowrap",
						color: "white",
					}}
				>
					Feature {feature}
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
					position: "relative",
					marginTop: "6px",
					minHeight: "30px", // Adjust this value as needed
					height: "auto",
				}}
				onClick={() => {
					console.log("clicked");
					setShowTest(true);
					setTimeout(() => {
						testTextRef.current?.focus();
					}, 100);
				}}
			>
				<div
					ref={testTextRef}
					contentEditable={true}
					style={{
						width: "100%",
						maxWidth: "100%",
						minWidth: "100%",
						borderRadius: "4px",
						fontSize: "12px",
						padding: "3px",
						fontFamily: "Inter",
						textAlign: "left",
						lineHeight: "1.5",
						color: "black",
						cursor: "text",
						border: "0px solid transparent",
						minHeight: "24px",
						outline: "none",
						whiteSpace: "pre-wrap",
						overflowWrap: "break-word",
						display: showTest ? "block" : "none",
					}}
					onFocus={() => setShowTest(true)}
					onBlur={() => setShowTest(false)}
					onInput={handleInput}
					onKeyDown={(e) => {
						if (e.key === "Enter" && !e.shiftKey) {
							e.preventDefault();
							submitTest(e);
						}
					}}
				/>
				{testActivations && textTokens && !showTest && (
					<div
						style={{
							width: "100%",
							maxWidth: "100%",
							minWidth: "100%",
							borderRadius: "4px",
							fontSize: "12px",
							padding: "3px",
							fontFamily: "Inter",
							textAlign: "left",
							lineHeight: "1.5",
							color: "black",
							minHeight: "24px",
							whiteSpace: "pre-wrap",
							overflowWrap: "break-word",
							position: "absolute",
							userSelect: "none",

							top: 0,
							left: 0,
							right: 0,
							bottom: 0,
							zIndex: 1,
						}}
					>
						<div style={{ display: "inline-block" }}>
							{textTokens.map((token: string, index: number) => {
								return (
									<TokenDisplay
										key={index}
										index={index}
										token={token}
										value={testActivations[0][index.toString()][0]}
										maxValue={maxAct}
									/>
								);
							})}
						</div>
					</div>
				)}
				{showTest && (
					<div
						style={{
							position: "absolute",
							bottom: "3px",
							right: "3px",
							cursor: "pointer",
							fontSize: "12px",
							color: "rgba(0, 0, 0, 0.5)",
						}}
						onClick={() => testTextRef.current?.blur()}
					>
						{loading ? (
							<svg
								width="16"
								height="16"
								viewBox="0 0 16 16"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
							>
								<path
									d="M8 1.5V4.5M8 11.5V14.5M3.5 8H0.5M15.5 8H12.5M13.3 13.3L11.1 11.1M13.3 2.7L11.1 4.9M2.7 13.3L4.9 11.1M2.7 2.7L4.9 4.9"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
									strokeLinejoin="round"
								>
									<animateTransform
										attributeName="transform"
										type="rotate"
										from="0 8 8"
										to="360 8 8"
										dur="1s"
										repeatCount="indefinite"
									/>
								</path>
							</svg>
						) : (
							<svg
								width="16"
								height="16"
								viewBox="0 0 16 16"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
							>
								<path
									d="M3 8H13M13 8L8 3M13 8L8 13"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
									strokeLinejoin="round"
								/>
							</svg>
						)}
					</div>
				)}
			</div>

			<div>
				{activations.slice(0, 3).map((activation: any, index: number) => (
					<ActivationItem key={index} activation={activation} maxAct={maxAct} />
				))}
			</div>

			<div
				ref={contentRef}
				style={{
					transition: "height 0.3s ease-in-out, opacity 0.5s ease-in-out",
					height: contentHeight,
					overflow: "hidden",
					opacity: opacity,
				}}
			>
				{activations.slice(3, 10).map((activation: any, index: number) => (
					<ActivationItem
						key={index + 3}
						activation={activation}
						maxAct={maxAct}
					/>
				))}
			</div>

			{activations.length > 3 && (
				<SampleToggle expanded={expanded} setExpanded={setExpanded} />
			)}
		</div>
	);
}

export default FeatureCard;
