import { DeleteIcon, MagnifyIcon } from "./Icons";

export const FeatureCardCommands = ({
	onDelete,
	onMagnify,
	featureId,
}: {
	onDelete?: (id: string) => void;
	onMagnify?: (id: string) => void;
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
			{onMagnify && <MagnifyIcon onClick={() => onMagnify(featureId)} />}
			{onDelete && <DeleteIcon onClick={() => onDelete(featureId)} />}
		</div>
	);
};
