import { NextRequest, NextResponse } from 'next/server';

const CODEWORDS_RUNTIME_URI = process.env.CODEWORDS_RUNTIME_URI!;
const CODEWORDS_API_KEY = process.env.CODEWORDS_API_KEY!;
const SERVICE_ID = 'smart_parking_analyzer_68f903f9';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const response = await fetch(
      `${CODEWORDS_RUNTIME_URI}/run/${SERVICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${CODEWORDS_API_KEY}`,
        },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${response.status}`, details: errorText },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to analyze parking image', details: String(error) },
      { status: 500 }
    );
  }
}
