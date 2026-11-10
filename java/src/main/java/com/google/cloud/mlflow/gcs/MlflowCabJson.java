/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.cloud.mlflow.gcs;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal, dependency-free JSON reader and string encoder.
 *
 * <p>The provider runs inside arbitrary Hadoop and Spark classpaths, so it deliberately avoids
 * depending on Jackson or Gson versions that differ across runtimes and shading layouts.
 */
final class MlflowCabJson {

  private final String text;
  private int pos;

  private MlflowCabJson(String text) {
    this.text = text;
  }

  /** Parses a JSON document into maps, lists, strings, numbers, booleans, and nulls. */
  static Object parse(String text) {
    if (text == null) {
      throw new IllegalArgumentException("JSON document is null");
    }
    MlflowCabJson parser = new MlflowCabJson(text);
    parser.skipWhitespace();
    Object value = parser.readValue();
    parser.skipWhitespace();
    if (parser.pos != text.length()) {
      throw new IllegalArgumentException("Unexpected trailing characters in JSON document");
    }
    return value;
  }

  /** Parses a JSON object, failing if the document is not an object. */
  @SuppressWarnings("unchecked")
  static Map<String, Object> parseObject(String text) {
    Object value = parse(text);
    if (!(value instanceof Map)) {
      throw new IllegalArgumentException("JSON document is not an object");
    }
    return (Map<String, Object>) value;
  }

  /** Returns the nested object stored under {@code key}, or null. */
  @SuppressWarnings("unchecked")
  static Map<String, Object> object(Map<String, Object> parent, String key) {
    Object value = parent == null ? null : parent.get(key);
    return value instanceof Map ? (Map<String, Object>) value : null;
  }

  /** Returns the first non-empty string value among {@code keys}. */
  static String firstString(Map<String, Object> parent, String... keys) {
    if (parent == null) {
      return null;
    }
    for (String key : keys) {
      Object value = parent.get(key);
      if (value instanceof String && !((String) value).isEmpty()) {
        return (String) value;
      }
    }
    return null;
  }

  /** Returns the first numeric value among {@code keys}, or null. */
  static Number firstNumber(Map<String, Object> parent, String... keys) {
    if (parent == null) {
      return null;
    }
    for (String key : keys) {
      Object value = parent.get(key);
      if (value instanceof Number) {
        return (Number) value;
      }
      if (value instanceof String) {
        try {
          return Long.parseLong(((String) value).trim());
        } catch (NumberFormatException ignored) {
          // Not a numeric string; keep looking.
        }
      }
    }
    return null;
  }

  /** Encodes {@code value} as a JSON string literal. */
  static String quote(String value) {
    StringBuilder out = new StringBuilder(value.length() + 2);
    out.append('"');
    for (int i = 0; i < value.length(); i++) {
      char ch = value.charAt(i);
      switch (ch) {
        case '"':
          out.append("\\\"");
          break;
        case '\\':
          out.append("\\\\");
          break;
        case '\n':
          out.append("\\n");
          break;
        case '\r':
          out.append("\\r");
          break;
        case '\t':
          out.append("\\t");
          break;
        case '\b':
          out.append("\\b");
          break;
        case '\f':
          out.append("\\f");
          break;
        default:
          if (ch < 0x20) {
            out.append(String.format("\\u%04x", (int) ch));
          } else {
            out.append(ch);
          }
      }
    }
    out.append('"');
    return out.toString();
  }

  /** Encodes a flat string-to-string map as a JSON object with stable key order. */
  static String encodeFlat(Map<String, String> values) {
    StringBuilder out = new StringBuilder("{");
    boolean first = true;
    for (Map.Entry<String, String> entry : values.entrySet()) {
      if (!first) {
        out.append(',');
      }
      first = false;
      out.append(quote(entry.getKey())).append(':');
      out.append(entry.getValue() == null ? "null" : quote(entry.getValue()));
    }
    return out.append('}').toString();
  }

  private Object readValue() {
    if (pos >= text.length()) {
      throw new IllegalArgumentException("Unexpected end of JSON document");
    }
    char ch = text.charAt(pos);
    switch (ch) {
      case '{':
        return readObject();
      case '[':
        return readArray();
      case '"':
        return readString();
      case 't':
        expectLiteral("true");
        return Boolean.TRUE;
      case 'f':
        expectLiteral("false");
        return Boolean.FALSE;
      case 'n':
        expectLiteral("null");
        return null;
      default:
        if (ch == '-' || (ch >= '0' && ch <= '9')) {
          return readNumber();
        }
        throw new IllegalArgumentException("Unexpected character in JSON document");
    }
  }

  private Map<String, Object> readObject() {
    Map<String, Object> result = new LinkedHashMap<>();
    pos++;
    skipWhitespace();
    if (peek() == '}') {
      pos++;
      return result;
    }
    while (true) {
      skipWhitespace();
      if (peek() != '"') {
        throw new IllegalArgumentException("Expected JSON object key");
      }
      String key = readString();
      skipWhitespace();
      if (peek() != ':') {
        throw new IllegalArgumentException("Expected ':' in JSON object");
      }
      pos++;
      skipWhitespace();
      result.put(key, readValue());
      skipWhitespace();
      char next = peek();
      pos++;
      if (next == '}') {
        return result;
      }
      if (next != ',') {
        throw new IllegalArgumentException("Expected ',' or '}' in JSON object");
      }
    }
  }

  private List<Object> readArray() {
    List<Object> result = new ArrayList<>();
    pos++;
    skipWhitespace();
    if (peek() == ']') {
      pos++;
      return result;
    }
    while (true) {
      skipWhitespace();
      result.add(readValue());
      skipWhitespace();
      char next = peek();
      pos++;
      if (next == ']') {
        return result;
      }
      if (next != ',') {
        throw new IllegalArgumentException("Expected ',' or ']' in JSON array");
      }
    }
  }

  private String readString() {
    StringBuilder out = new StringBuilder();
    pos++;
    while (true) {
      if (pos >= text.length()) {
        throw new IllegalArgumentException("Unterminated JSON string");
      }
      char ch = text.charAt(pos++);
      if (ch == '"') {
        return out.toString();
      }
      if (ch != '\\') {
        out.append(ch);
        continue;
      }
      if (pos >= text.length()) {
        throw new IllegalArgumentException("Unterminated JSON escape");
      }
      char esc = text.charAt(pos++);
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.append(esc);
          break;
        case 'b':
          out.append('\b');
          break;
        case 'f':
          out.append('\f');
          break;
        case 'n':
          out.append('\n');
          break;
        case 'r':
          out.append('\r');
          break;
        case 't':
          out.append('\t');
          break;
        case 'u':
          if (pos + 4 > text.length()) {
            throw new IllegalArgumentException("Invalid JSON unicode escape");
          }
          out.append((char) Integer.parseInt(text.substring(pos, pos + 4), 16));
          pos += 4;
          break;
        default:
          throw new IllegalArgumentException("Invalid JSON escape");
      }
    }
  }

  private Number readNumber() {
    int start = pos;
    if (peek() == '-') {
      pos++;
    }
    boolean isDecimal = false;
    while (pos < text.length()) {
      char ch = text.charAt(pos);
      if (ch >= '0' && ch <= '9') {
        pos++;
      } else if (ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-') {
        isDecimal = true;
        pos++;
      } else {
        break;
      }
    }
    String literal = text.substring(start, pos);
    if (!isDecimal) {
      try {
        return Long.parseLong(literal);
      } catch (NumberFormatException ignored) {
        // Fall through to double for out-of-range integers.
      }
    }
    return Double.parseDouble(literal);
  }

  private void expectLiteral(String literal) {
    if (!text.startsWith(literal, pos)) {
      throw new IllegalArgumentException("Invalid JSON literal");
    }
    pos += literal.length();
  }

  private char peek() {
    if (pos >= text.length()) {
      throw new IllegalArgumentException("Unexpected end of JSON document");
    }
    return text.charAt(pos);
  }

  private void skipWhitespace() {
    while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) {
      pos++;
    }
  }
}
