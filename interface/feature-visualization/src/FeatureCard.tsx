import { useState, useEffect, useRef } from "react";
import "./App.css";
// import { keyframes } from "@emotion/react";
import { FeatureItem } from "./types";

const getBaseUrl = () => {
	return process.env.NODE_ENV === "development"
		? "http://localhost:5000"
		: "https://steering-explorer-server.vercel.app";
};

const TokenDisplay = ({
	token,
	value,
	maxValue,
}: {
	token: string;
	value: number;
	maxValue: number;
}) => {
	const opacity = Math.min(0.85, value / maxValue);
	const [isHovering, setIsHovering] = useState(false);
	const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
	const spanRef = useRef<HTMLSpanElement>(null);

	// Remove "▁" if it's the first character, otherwise keep the token as is
	const displayToken = token.startsWith("▁") ? token.slice(1) : token;
	const addSpace = token.includes("▁") ? " " : "";

	const updateTooltipPosition = () => {
		if (spanRef.current) {
			const rect = spanRef.current.getBoundingClientRect();
			setTooltipPosition({
				top: rect.top - 24, // 24px above the span
				left: rect.left + rect.width / 2,
			});
		}
	};

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
		>
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
					backgroundColor: `rgba(42, 97, 211, ${opacity})`,
					display: "inline-block",
					borderRadius: "4px",
					color: "black",
					// color: opacity > 0.6 ? "white" : "black",
				}}
			>
				{addSpace}
				{displayToken}
			</span>
		</span>
	);
};

const ActivationItem = ({ activation }: { activation: any }) => {
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
						token={token}
						value={activation.values[i + startIndex]}
						maxValue={Math.max(...activation.values)}
					/>
				))}
			</div>
		</div>
	);
};

const FeatureCardCommands = ({
	columnSide,
	inspectFeature,
	inspectedFeature,
	onDelete,
	feature,
	featureId,
	peek,
	hide,
}: {
	columnSide: "left" | "right";
	inspectFeature: (feature: FeatureItem) => void;
	inspectedFeature: FeatureItem | null;
	onDelete: (id: string) => void;
	feature: number;
	featureId: string;
	peek: () => void;
	hide: () => void;
}) => {
	return (
		<div>
			{columnSide === "left" ? (
				<>
					{/* <div
						style={{
							position: "absolute",
							top: "7px",
							right: "56px",
							cursor: "pointer",
							fontSize: "16px",
							color: "gray",
							transition: "color 0.1s ease-in-out",
						}}
						onMouseDown={() => {
							peek();
						}}
						onMouseUp={() => {
							hide();
						}}
						onMouseEnter={(e) => (e.currentTarget.style.color = "black")}
						onMouseLeave={(e) => {
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
								d="M1 8C1 8 3.5 3 8 3C12.5 3 15 8 15 8C15 8 12.5 13 8 13C3.5 13 1 8 1 8Z"
								stroke="currentColor"
								strokeWidth="1.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
							<path
								d="M8 10C9.10457 10 10 9.10457 10 8C10 6.89543 9.10457 6 8 6C6.89543 6 6 6.89543 6 8C6 9.10457 6.89543 10 8 10Z"
								stroke="currentColor"
								strokeWidth="1.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
						</svg>
					</div> */}
					<div
						style={{
							position: "absolute",
							top: "7px",
							right: "33px",
							cursor: "pointer",
							fontSize: "16px",
							// color: inspectedFeature?.id === feature.id ? "black" : "gray",
							transition: "color 0.1s ease-in-out",
						}}
						// onClick={() => {
						// 	inspectFeature(feature);
						// }}
						onMouseEnter={(e) => (e.currentTarget.style.color = "black")}
						onMouseLeave={(e) => {
							e.currentTarget.style.color = "gray";
							// e.currentTarget.style.color =
							// 	inspectedFeature?.id === feature.id ? "black" : "gray";
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
								d="M7.33333 12.6667C10.2789 12.6667 12.6667 10.2789 12.6667 7.33333C12.6667 4.38781 10.2789 2 7.33333 2C4.38781 2 2 4.38781 2 7.33333C2 10.2789 4.38781 12.6667 7.33333 12.6667Z"
								stroke="currentColor"
								strokeWidth="1.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
							<path
								d="M14 14L11 11"
								stroke="currentColor"
								strokeWidth="1.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
						</svg>
					</div>
					<div
						style={{
							position: "absolute",
							top: "7px",
							right: "10px",
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
				</>
			) : (
				<>
					{" "}
					<div
						style={{
							position: "absolute",
							top: "7px",
							right: "10px",
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
								d="M8 3v10M3 8h10"
								stroke="currentColor"
								strokeWidth="1.5"
								strokeLinecap="round"
								strokeLinejoin="round"
							/>
						</svg>
					</div>
				</>
			)}
		</div>
	);
};

function FeatureCard({
	feature,
	featureId,
	onDelete,
	inspectFeature,
	inspectedFeature,
	columnSide,
	activations = [],
}: {
	feature: number;
	featureId: string;
	onDelete: (id: string) => void;
	inspectFeature: (feature: FeatureItem) => void;
	inspectedFeature: FeatureItem | null;
	columnSide: "left" | "right";
	activations: any;
}) {
	// const [activations, setActivations] = useState([]);
	const [description, setDescription] = useState("");
	const [expanded, setExpanded] = useState(false);
	const [contentHeight, setContentHeight] = useState("auto");
	const contentRef = useRef<HTMLDivElement>(null);
	const [opacity, setOpacity] = useState(0);

	const [loading, setLoading] = useState(true);

	// useEffect(() => {
	// 	fetch(
	// 		`https://siunami--steering-webapp-analyze-activations-dev.modal.run?feature=${4000}`
	// 	).then((data) => {
	// 		console.log(data);
	// 	});
	// }, []);

	// const fetchFeatureData = async () => {
	// 	try {
	// 		const response = await fetch(
	// 			// `${getBaseUrl()}/api/feature/${feature.featureNumber}`,
	// 			`${getBaseUrl()}/get_feature?feature=${feature}`,
	// 			{
	// 				method: "GET",
	// 				headers: {
	// 					"Content-Type": "application/json",
	// 				},
	// 			}
	// 		);
	// 		const data = await response.json();
	// 		console.log(data);
	// 		// Filter out duplicate activations
	// 		const uniqueActivations = data.activations.filter(
	// 			(activation: any, index: number, self: any) =>
	// 				index ===
	// 				self.findIndex(
	// 					(t: any) => t.tokens.join("") === activation.tokens.join("")
	// 				)
	// 		);
	// 		setLoading(false);
	// 		setActivations(uniqueActivations);
	// 		// setDescription(data.explanations[0].description);
	// 		setDescription("");
	// 	} catch (error) {
	// 		console.error("Error fetching feature data:", error);
	// 	}
	// };

	// useEffect(() => {
	// 	fetchFeatureData();
	// }, [feature]);

	useEffect(() => {
		setLoading(false);
	}, [activations]);

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

	const peek = () => {
		setExpanded(true);
	};

	const hide = () => {
		setExpanded(false);
	};

	return (
		<div
			style={{
				// backgroundColor:
				// 	inspectedFeature?.id === feature.id
				// 		? "rgba(250, 250, 248, 1)"
				// 		: "rgba(250, 250, 248, 1)",
				// border:
				// 	inspectedFeature?.id === feature.id
				// 		? "6px solid rgba(0, 0, 255, 0.65)"
				// 		: "6px solid rgba(0, 0, 0, 0)",
				backgroundColor: "rgba(250, 250, 248, 1)",
				border: "6px solid rgba(0, 0, 0, 0)",
				padding: "8px",
				paddingBottom: "4px",
				borderRadius: "15px",
				margin: "0 12px",
				// minWidth: "400px",
				textAlign: "center",
				width: "calc(100% - 24px)",
				maxWidth: columnSide === "left" ? "650px" : "450px",
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
					columnSide={columnSide}
					inspectFeature={inspectFeature}
					inspectedFeature={inspectedFeature}
					onDelete={onDelete}
					feature={feature}
					featureId={featureId}
					peek={peek}
					hide={hide}
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
						marginRight: columnSide === "left" ? "45px" : "33px",
						padding: "1px",
						fontWeight: "bold",
						textAlign: "left",
						color: "rgba(0, 0, 0, 0.5)",
					}}
				>
					{description}
					{loading && <span className="loading-text">Loading</span>}
					{activations.length == 0 && !loading && "No activations found"}
				</div>
			</div>

			<div>
				{activations.slice(0, 3).map((activation: any, index: number) => (
					<ActivationItem key={index} activation={activation} />
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
					<ActivationItem key={index + 3} activation={activation} />
				))}
			</div>

			{activations.length > 3 && (
				<div
					onClick={() => setExpanded(!expanded)}
					style={{
						// marginTop: "10px",
						// padding: "5px",
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
			)}
		</div>
	);
}

export default FeatureCard;
