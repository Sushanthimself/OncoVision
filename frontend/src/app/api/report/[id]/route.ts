import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  const res = await fetch(`${API_URL}/report/${id}/pdf`);
  if (!res.ok) {
    return NextResponse.json({ error: "PDF generation failed" }, { status: 502 });
  }

  const pdfBuffer = await res.arrayBuffer();

  return new NextResponse(pdfBuffer, {
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename="OncoVision_Report_${id.slice(0, 8)}.pdf"`,
    },
  });
}
