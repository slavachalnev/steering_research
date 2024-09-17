import React, { useState } from "react";

interface IconProps {
	onClick?: () => void;
	style?: React.CSSProperties;
	path: string;
	children?: React.ReactNode; // Add this line
}

const BaseIcon: React.FC<IconProps> = ({ onClick, style, path, children }) => {
	const [isHovered, setIsHovered] = useState(false);

	return (
		<svg
			width="16"
			height="16"
			viewBox="0 0 16 16"
			fill="none"
			xmlns="http://www.w3.org/2000/svg"
			onMouseEnter={() => setIsHovered(true)}
			onMouseLeave={() => setIsHovered(false)}
			onClick={onClick}
			style={Object.assign(
				{},
				{ cursor: "pointer", color: isHovered ? "black" : "grey" },
				style
			)}
		>
			<path
				d={path}
				stroke="currentColor"
				strokeWidth="1.5"
				strokeLinecap="round"
				strokeLinejoin="round"
			/>
			{children} {/* Add this line */}
		</svg>
	);
};

export const MinusIcon: React.FC<Omit<IconProps, "path">> = (props) => (
	<BaseIcon {...props} path="M3 8H13" />
);

export const PlusIcon: React.FC<Omit<IconProps, "path">> = (props) => (
	<BaseIcon {...props} path="M8 3V13M3 8H13" />
);

export const RightArrowIcon: React.FC<Omit<IconProps, "path">> = (props) => (
	<BaseIcon {...props} path="M3 8H13M13 8L8 3M13 8L8 13" />
);

export const LoadingIcon: React.FC<Omit<IconProps, "path">> = (props) => (
	<BaseIcon {...props} path="">
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
	</BaseIcon>
);

export const MagnifyIcon: React.FC<Omit<IconProps, "path">> = (props) => (
	<BaseIcon
		{...props}
		path="M7 11.5C9.48528 11.5 11.5 9.48528 11.5 7C11.5 4.51472 9.48528 2.5 7 2.5C4.51472 2.5 2.5 4.51472 2.5 7C2.5 9.48528 4.51472 11.5 7 11.5Z M10.5 10.5L13.5 13.5"
	/>
);

export const DeleteIcon: React.FC<Omit<IconProps, "path">> = (props) => (
	<BaseIcon
		{...props}
		path="M2 4h12M5.333 4V2.667a1.333 1.333 0 011.334-1.334h2.666a1.333 1.333 0 011.334 1.334V4m2 0v9.333a1.333 1.333 0 01-1.334 1.334H4.667a1.333 1.333 0 01-1.334-1.334V4h9.334z"
	/>
);
