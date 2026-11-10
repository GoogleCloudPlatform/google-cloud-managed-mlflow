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

import java.util.regex.Pattern;

/** Scrubs bearer credentials from strings before they reach logs, exceptions, or results. */
public final class MlflowCabRedaction {

  private static final String REDACTED = "<redacted>";
  private static final Pattern OAUTH_TOKEN = Pattern.compile("ya29\\.[A-Za-z0-9._\\-~+/=]*");
  private static final Pattern BEARER =
      Pattern.compile("(?i)(bearer)\\s+[A-Za-z0-9._\\-~+/=]+");
  private static final Pattern JSON_SECRET =
      Pattern.compile(
          "(?i)(\"(?:access_?token|refresh_?token|id_?token|client_?secret|private_?key|token)\""
              + "\\s*:\\s*)\"[^\"]*\"");
  private static final Pattern FORM_SECRET =
      Pattern.compile("(?i)((?:access_token|refresh_token|assertion|client_secret)=)[^&\\s]+");

  private MlflowCabRedaction() {}

  /** Returns {@code value} with OAuth tokens and secret JSON/form fields replaced. */
  public static String redact(String value) {
    if (value == null || value.isEmpty()) {
      return value;
    }
    String out = OAUTH_TOKEN.matcher(value).replaceAll(REDACTED);
    out = BEARER.matcher(out).replaceAll("$1 " + REDACTED);
    out = JSON_SECRET.matcher(out).replaceAll("$1\"" + REDACTED + "\"");
    out = FORM_SECRET.matcher(out).replaceAll("$1" + REDACTED);
    return out;
  }

  /** Returns a redacted, length-bounded message for {@code error}. */
  public static String describe(Throwable error) {
    if (error == null) {
      return "";
    }
    String message = error.getMessage();
    String text = error.getClass().getSimpleName() + (message == null ? "" : ": " + message);
    text = redact(text);
    return text.length() > 600 ? text.substring(0, 600) + "..." : text;
  }
}
