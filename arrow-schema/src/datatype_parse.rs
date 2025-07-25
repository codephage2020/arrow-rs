// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{fmt::Display, iter::Peekable, str::Chars, sync::Arc};

use crate::{ArrowError, DataType, Field, Fields, IntervalUnit, TimeUnit};

pub(crate) fn parse_data_type(val: &str) -> ArrowResult<DataType> {
    Parser::new(val).parse()
}

type ArrowResult<T> = Result<T, ArrowError>;

fn make_error(val: &str, msg: &str) -> ArrowError {
    let msg = format!("Unsupported type '{val}'. Must be a supported arrow type name such as 'Int32' or 'Timestamp(Nanosecond, None)'. Error {msg}" );
    ArrowError::ParseError(msg)
}

fn make_error_expected(val: &str, expected: &Token, actual: &Token) -> ArrowError {
    make_error(val, &format!("Expected '{expected}', got '{actual}'"))
}

#[derive(Debug)]
/// Implementation of `parse_data_type`, modeled after <https://github.com/sqlparser-rs/sqlparser-rs>
struct Parser<'a> {
    val: &'a str,
    tokenizer: Tokenizer<'a>,
}

impl<'a> Parser<'a> {
    fn new(val: &'a str) -> Self {
        Self {
            val,
            tokenizer: Tokenizer::new(val),
        }
    }

    fn parse(mut self) -> ArrowResult<DataType> {
        let data_type = self.parse_next_type()?;
        // ensure that there is no trailing content
        if self.tokenizer.next().is_some() {
            Err(make_error(
                self.val,
                &format!("checking trailing content after parsing '{data_type}'"),
            ))
        } else {
            Ok(data_type)
        }
    }

    /// parses the next full DataType
    fn parse_next_type(&mut self) -> ArrowResult<DataType> {
        match self.next_token()? {
            Token::SimpleType(data_type) => Ok(data_type),
            Token::Timestamp => self.parse_timestamp(),
            Token::Time32 => self.parse_time32(),
            Token::Time64 => self.parse_time64(),
            Token::Duration => self.parse_duration(),
            Token::Interval => self.parse_interval(),
            Token::FixedSizeBinary => self.parse_fixed_size_binary(),
            Token::Decimal32 => self.parse_decimal_32(),
            Token::Decimal64 => self.parse_decimal_64(),
            Token::Decimal128 => self.parse_decimal_128(),
            Token::Decimal256 => self.parse_decimal_256(),
            Token::Dictionary => self.parse_dictionary(),
            Token::List => self.parse_list(),
            Token::LargeList => self.parse_large_list(),
            Token::FixedSizeList => self.parse_fixed_size_list(),
            Token::Struct => self.parse_struct(),
            Token::FieldName(word) => {
                Err(make_error(self.val, &format!("unrecognized word: {word}")))
            }
            tok => Err(make_error(
                self.val,
                &format!("finding next type, got unexpected '{tok}'"),
            )),
        }
    }

    /// Parses the List type
    fn parse_list(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let data_type = self.parse_next_type()?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::List(Arc::new(Field::new_list_field(
            data_type, true,
        ))))
    }

    /// Parses the LargeList type
    fn parse_large_list(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let data_type = self.parse_next_type()?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::LargeList(Arc::new(Field::new_list_field(
            data_type, true,
        ))))
    }

    /// Parses the FixedSizeList type
    fn parse_fixed_size_list(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let length = self.parse_i32("FixedSizeList")?;
        self.expect_token(Token::Comma)?;
        let data_type = self.parse_next_type()?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::FixedSizeList(
            Arc::new(Field::new_list_field(data_type, true)),
            length,
        ))
    }

    /// Parses the next timeunit
    fn parse_time_unit(&mut self, context: &str) -> ArrowResult<TimeUnit> {
        match self.next_token()? {
            Token::TimeUnit(time_unit) => Ok(time_unit),
            tok => Err(make_error(
                self.val,
                &format!("finding TimeUnit for {context}, got {tok}"),
            )),
        }
    }

    /// Parses the next timezone
    fn parse_timezone(&mut self, context: &str) -> ArrowResult<Option<String>> {
        match self.next_token()? {
            Token::None => Ok(None),
            Token::Some => {
                self.expect_token(Token::LParen)?;
                let timezone = self.parse_double_quoted_string("Timezone")?;
                self.expect_token(Token::RParen)?;
                Ok(Some(timezone))
            }
            tok => Err(make_error(
                self.val,
                &format!("finding Timezone for {context}, got {tok}"),
            )),
        }
    }

    /// Parses the next double quoted string
    fn parse_double_quoted_string(&mut self, context: &str) -> ArrowResult<String> {
        match self.next_token()? {
            Token::DoubleQuotedString(s) => Ok(s),
            Token::FieldName(word) => {
                Err(make_error(self.val, &format!("unrecognized word: {word}")))
            }
            tok => Err(make_error(
                self.val,
                &format!("finding double quoted string for {context}, got '{tok}'"),
            )),
        }
    }

    /// Parses the next integer value
    fn parse_i64(&mut self, context: &str) -> ArrowResult<i64> {
        match self.next_token()? {
            Token::Integer(v) => Ok(v),
            tok => Err(make_error(
                self.val,
                &format!("finding i64 for {context}, got '{tok}'"),
            )),
        }
    }

    /// Parses the next i32 integer value
    fn parse_i32(&mut self, context: &str) -> ArrowResult<i32> {
        let length = self.parse_i64(context)?;
        length.try_into().map_err(|e| {
            make_error(
                self.val,
                &format!("converting {length} into i32 for {context}: {e}"),
            )
        })
    }

    /// Parses the next i8 integer value
    fn parse_i8(&mut self, context: &str) -> ArrowResult<i8> {
        let length = self.parse_i64(context)?;
        length.try_into().map_err(|e| {
            make_error(
                self.val,
                &format!("converting {length} into i8 for {context}: {e}"),
            )
        })
    }

    /// Parses the next u8 integer value
    fn parse_u8(&mut self, context: &str) -> ArrowResult<u8> {
        let length = self.parse_i64(context)?;
        length.try_into().map_err(|e| {
            make_error(
                self.val,
                &format!("converting {length} into u8 for {context}: {e}"),
            )
        })
    }

    /// Parses the next timestamp (called after `Timestamp` has been consumed)
    fn parse_timestamp(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let time_unit = self.parse_time_unit("Timestamp")?;
        self.expect_token(Token::Comma)?;
        let timezone = self.parse_timezone("Timestamp")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Timestamp(time_unit, timezone.map(Into::into)))
    }

    /// Parses the next Time32 (called after `Time32` has been consumed)
    fn parse_time32(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let time_unit = self.parse_time_unit("Time32")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Time32(time_unit))
    }

    /// Parses the next Time64 (called after `Time64` has been consumed)
    fn parse_time64(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let time_unit = self.parse_time_unit("Time64")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Time64(time_unit))
    }

    /// Parses the next Duration (called after `Duration` has been consumed)
    fn parse_duration(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let time_unit = self.parse_time_unit("Duration")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Duration(time_unit))
    }

    /// Parses the next Interval (called after `Interval` has been consumed)
    fn parse_interval(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let interval_unit = match self.next_token()? {
            Token::IntervalUnit(interval_unit) => interval_unit,
            tok => {
                return Err(make_error(
                    self.val,
                    &format!("finding IntervalUnit for Interval, got {tok}"),
                ))
            }
        };
        self.expect_token(Token::RParen)?;
        Ok(DataType::Interval(interval_unit))
    }

    /// Parses the next FixedSizeBinary (called after `FixedSizeBinary` has been consumed)
    fn parse_fixed_size_binary(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let length = self.parse_i32("FixedSizeBinary")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::FixedSizeBinary(length))
    }

    /// Parses the next Decimal32 (called after `Decimal32` has been consumed)
    fn parse_decimal_32(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let precision = self.parse_u8("Decimal32")?;
        self.expect_token(Token::Comma)?;
        let scale = self.parse_i8("Decimal32")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Decimal32(precision, scale))
    }

    /// Parses the next Decimal64 (called after `Decimal64` has been consumed)
    fn parse_decimal_64(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let precision = self.parse_u8("Decimal64")?;
        self.expect_token(Token::Comma)?;
        let scale = self.parse_i8("Decimal64")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Decimal64(precision, scale))
    }

    /// Parses the next Decimal128 (called after `Decimal128` has been consumed)
    fn parse_decimal_128(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let precision = self.parse_u8("Decimal128")?;
        self.expect_token(Token::Comma)?;
        let scale = self.parse_i8("Decimal128")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Decimal128(precision, scale))
    }

    /// Parses the next Decimal256 (called after `Decimal256` has been consumed)
    fn parse_decimal_256(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let precision = self.parse_u8("Decimal256")?;
        self.expect_token(Token::Comma)?;
        let scale = self.parse_i8("Decimal256")?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Decimal256(precision, scale))
    }

    /// Parses the next Dictionary (called after `Dictionary` has been consumed)
    fn parse_dictionary(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let key_type = self.parse_next_type()?;
        self.expect_token(Token::Comma)?;
        let value_type = self.parse_next_type()?;
        self.expect_token(Token::RParen)?;
        Ok(DataType::Dictionary(
            Box::new(key_type),
            Box::new(value_type),
        ))
    }
    fn parse_struct(&mut self) -> ArrowResult<DataType> {
        self.expect_token(Token::LParen)?;
        let mut fields = Vec::new();
        loop {
            let field_name = match self.next_token()? {
                // It's valid to have a name that is a type name
                Token::SimpleType(data_type) => data_type.to_string(),
                Token::FieldName(name) => name,
                Token::RParen => {
                    if fields.is_empty() {
                        break;
                    } else {
                        return Err(make_error(
                            self.val,
                            "Unexpected token while parsing Struct fields. Expected a word for the name of Struct, but got trailing comma",
                        ));
                    }
                }
                tok => {
                    return Err(make_error(
                        self.val,
                        &format!("Expected a word for the name of Struct, but got {tok}"),
                    ))
                }
            };
            let field_type = self.parse_next_type()?;
            fields.push(Arc::new(Field::new(field_name, field_type, true)));
            match self.next_token()? {
                Token::Comma => continue,
                Token::RParen => break,
                tok => {
                    return Err(make_error(
                        self.val,
                        &format!("Unexpected token while parsing Struct fields. Expected ',' or ')', but got '{tok}'"),
                    ))
                }
            }
        }
        Ok(DataType::Struct(Fields::from(fields)))
    }

    /// return the next token, or an error if there are none left
    fn next_token(&mut self) -> ArrowResult<Token> {
        match self.tokenizer.next() {
            None => Err(make_error(self.val, "finding next token")),
            Some(token) => token,
        }
    }

    /// consume the next token, returning OK(()) if it matches tok, and Err if not
    fn expect_token(&mut self, tok: Token) -> ArrowResult<()> {
        let next_token = self.next_token()?;
        if next_token == tok {
            Ok(())
        } else {
            Err(make_error_expected(self.val, &tok, &next_token))
        }
    }
}

/// returns true if this character is a separator
fn is_separator(c: char) -> bool {
    c == '(' || c == ')' || c == ',' || c == ' '
}

#[derive(Debug)]
/// Splits a strings like Dictionary(Int32, Int64) into tokens sutable for parsing
///
/// For example the string "Timestamp(Nanosecond, None)" would be parsed into:
///
/// * Token::Timestamp
/// * Token::Lparen
/// * Token::IntervalUnit(IntervalUnit::Nanosecond)
/// * Token::Comma,
/// * Token::None,
/// * Token::Rparen,
struct Tokenizer<'a> {
    val: &'a str,
    chars: Peekable<Chars<'a>>,
    // temporary buffer for parsing words
    word: String,
}

impl<'a> Tokenizer<'a> {
    fn new(val: &'a str) -> Self {
        Self {
            val,
            chars: val.chars().peekable(),
            word: String::new(),
        }
    }

    /// returns the next char, without consuming it
    fn peek_next_char(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    /// returns the next char, and consuming it
    fn next_char(&mut self) -> Option<char> {
        self.chars.next()
    }

    /// parse the characters in val starting at pos, until the next
    /// `,`, `(`, or `)` or end of line
    fn parse_word(&mut self) -> ArrowResult<Token> {
        // reset temp space
        self.word.clear();
        loop {
            match self.peek_next_char() {
                None => break,
                Some(c) if is_separator(c) => break,
                Some(c) => {
                    self.next_char();
                    self.word.push(c);
                }
            }
        }

        if let Some(c) = self.word.chars().next() {
            // if it started with a number, try parsing it as an integer
            if c == '-' || c.is_numeric() {
                let val: i64 = self.word.parse().map_err(|e| {
                    make_error(self.val, &format!("parsing {} as integer: {e}", self.word))
                })?;
                return Ok(Token::Integer(val));
            }
            // if it started with a double quote `"`, try parsing it as a double quoted string
            else if c == '"' {
                let len = self.word.chars().count();

                // to verify it's double quoted
                if let Some(last_c) = self.word.chars().last() {
                    if last_c != '"' || len < 2 {
                        return Err(make_error(
                            self.val,
                            &format!(
                                "parsing {} as double quoted string: last char must be \"",
                                self.word
                            ),
                        ));
                    }
                }

                if len == 2 {
                    return Err(make_error(
                        self.val,
                        &format!(
                            "parsing {} as double quoted string: empty string isn't supported",
                            self.word
                        ),
                    ));
                }

                let val: String = self.word.parse().map_err(|e| {
                    make_error(
                        self.val,
                        &format!("parsing {} as double quoted string: {e}", self.word),
                    )
                })?;

                let s = val[1..len - 1].to_string();
                if s.contains('"') {
                    return Err(make_error(
                        self.val,
                        &format!("parsing {} as double quoted string: escaped double quote isn't supported", self.word),
                    ));
                }

                return Ok(Token::DoubleQuotedString(s));
            }
        }

        // figure out what the word was
        let token = match self.word.as_str() {
            "Null" => Token::SimpleType(DataType::Null),
            "Boolean" => Token::SimpleType(DataType::Boolean),

            "Int8" => Token::SimpleType(DataType::Int8),
            "Int16" => Token::SimpleType(DataType::Int16),
            "Int32" => Token::SimpleType(DataType::Int32),
            "Int64" => Token::SimpleType(DataType::Int64),

            "UInt8" => Token::SimpleType(DataType::UInt8),
            "UInt16" => Token::SimpleType(DataType::UInt16),
            "UInt32" => Token::SimpleType(DataType::UInt32),
            "UInt64" => Token::SimpleType(DataType::UInt64),

            "Utf8" => Token::SimpleType(DataType::Utf8),
            "LargeUtf8" => Token::SimpleType(DataType::LargeUtf8),
            "Utf8View" => Token::SimpleType(DataType::Utf8View),
            "Binary" => Token::SimpleType(DataType::Binary),
            "BinaryView" => Token::SimpleType(DataType::BinaryView),
            "LargeBinary" => Token::SimpleType(DataType::LargeBinary),

            "Float16" => Token::SimpleType(DataType::Float16),
            "Float32" => Token::SimpleType(DataType::Float32),
            "Float64" => Token::SimpleType(DataType::Float64),

            "Date32" => Token::SimpleType(DataType::Date32),
            "Date64" => Token::SimpleType(DataType::Date64),

            "List" => Token::List,
            "LargeList" => Token::LargeList,
            "FixedSizeList" => Token::FixedSizeList,

            "Second" => Token::TimeUnit(TimeUnit::Second),
            "Millisecond" => Token::TimeUnit(TimeUnit::Millisecond),
            "Microsecond" => Token::TimeUnit(TimeUnit::Microsecond),
            "Nanosecond" => Token::TimeUnit(TimeUnit::Nanosecond),

            "Timestamp" => Token::Timestamp,
            "Time32" => Token::Time32,
            "Time64" => Token::Time64,
            "Duration" => Token::Duration,
            "Interval" => Token::Interval,
            "Dictionary" => Token::Dictionary,

            "FixedSizeBinary" => Token::FixedSizeBinary,

            "Decimal32" => Token::Decimal32,
            "Decimal64" => Token::Decimal64,
            "Decimal128" => Token::Decimal128,
            "Decimal256" => Token::Decimal256,

            "YearMonth" => Token::IntervalUnit(IntervalUnit::YearMonth),
            "DayTime" => Token::IntervalUnit(IntervalUnit::DayTime),
            "MonthDayNano" => Token::IntervalUnit(IntervalUnit::MonthDayNano),

            "Some" => Token::Some,
            "None" => Token::None,

            "Struct" => Token::Struct,
            // If we don't recognize the word, treat it as a field name
            word => Token::FieldName(word.to_string()),
        };
        Ok(token)
    }
}

impl Iterator for Tokenizer<'_> {
    type Item = ArrowResult<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.peek_next_char()? {
                ' ' => {
                    // skip whitespace
                    self.next_char();
                    continue;
                }
                '(' => {
                    self.next_char();
                    return Some(Ok(Token::LParen));
                }
                ')' => {
                    self.next_char();
                    return Some(Ok(Token::RParen));
                }
                ',' => {
                    self.next_char();
                    return Some(Ok(Token::Comma));
                }
                _ => return Some(self.parse_word()),
            }
        }
    }
}

/// Grammar is
///
#[derive(Debug, PartialEq)]
enum Token {
    // Null, or Int32
    SimpleType(DataType),
    Timestamp,
    Time32,
    Time64,
    Duration,
    Interval,
    FixedSizeBinary,
    Decimal32,
    Decimal64,
    Decimal128,
    Decimal256,
    Dictionary,
    TimeUnit(TimeUnit),
    IntervalUnit(IntervalUnit),
    LParen,
    RParen,
    Comma,
    Some,
    None,
    Integer(i64),
    DoubleQuotedString(String),
    List,
    LargeList,
    FixedSizeList,
    Struct,
    FieldName(String),
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::SimpleType(t) => write!(f, "{t}"),
            Token::List => write!(f, "List"),
            Token::LargeList => write!(f, "LargeList"),
            Token::FixedSizeList => write!(f, "FixedSizeList"),
            Token::Timestamp => write!(f, "Timestamp"),
            Token::Time32 => write!(f, "Time32"),
            Token::Time64 => write!(f, "Time64"),
            Token::Duration => write!(f, "Duration"),
            Token::Interval => write!(f, "Interval"),
            Token::TimeUnit(u) => write!(f, "TimeUnit({u:?})"),
            Token::IntervalUnit(u) => write!(f, "IntervalUnit({u:?})"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Some => write!(f, "Some"),
            Token::None => write!(f, "None"),
            Token::FixedSizeBinary => write!(f, "FixedSizeBinary"),
            Token::Decimal32 => write!(f, "Decimal32"),
            Token::Decimal64 => write!(f, "Decimal64"),
            Token::Decimal128 => write!(f, "Decimal128"),
            Token::Decimal256 => write!(f, "Decimal256"),
            Token::Dictionary => write!(f, "Dictionary"),
            Token::Integer(v) => write!(f, "Integer({v})"),
            Token::DoubleQuotedString(s) => write!(f, "DoubleQuotedString({s})"),
            Token::Struct => write!(f, "Struct"),
            Token::FieldName(s) => write!(f, "FieldName({s})"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_data_type() {
        // this ensures types can be parsed correctly from their string representations
        for dt in list_datatypes() {
            round_trip(dt)
        }
    }

    /// convert data_type to a string, and then parse it as a type
    /// verifying it is the same
    fn round_trip(data_type: DataType) {
        let data_type_string = data_type.to_string();
        println!("Input '{data_type_string}' ({data_type:?})");
        let parsed_type = parse_data_type(&data_type_string).unwrap();
        assert_eq!(
            data_type, parsed_type,
            "Mismatch parsing {data_type_string}"
        );
    }

    fn list_datatypes() -> Vec<DataType> {
        vec![
            // ---------
            // Non Nested types
            // ---------
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            // we can't cover all possible timezones, here we only test utc and +08:00
            DataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".into())),
            DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
            DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
            DataType::Timestamp(TimeUnit::Second, Some("+00:00".into())),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("+08:00".into())),
            DataType::Timestamp(TimeUnit::Microsecond, Some("+08:00".into())),
            DataType::Timestamp(TimeUnit::Millisecond, Some("+08:00".into())),
            DataType::Timestamp(TimeUnit::Second, Some("+08:00".into())),
            DataType::Date32,
            DataType::Date64,
            DataType::Time32(TimeUnit::Second),
            DataType::Time32(TimeUnit::Millisecond),
            DataType::Time32(TimeUnit::Microsecond),
            DataType::Time32(TimeUnit::Nanosecond),
            DataType::Time64(TimeUnit::Second),
            DataType::Time64(TimeUnit::Millisecond),
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Duration(TimeUnit::Second),
            DataType::Duration(TimeUnit::Millisecond),
            DataType::Duration(TimeUnit::Microsecond),
            DataType::Duration(TimeUnit::Nanosecond),
            DataType::Interval(IntervalUnit::YearMonth),
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Binary,
            DataType::BinaryView,
            DataType::FixedSizeBinary(0),
            DataType::FixedSizeBinary(1234),
            DataType::FixedSizeBinary(-432),
            DataType::LargeBinary,
            DataType::Utf8,
            DataType::Utf8View,
            DataType::LargeUtf8,
            DataType::Decimal32(7, 8),
            DataType::Decimal64(6, 9),
            DataType::Decimal128(7, 12),
            DataType::Decimal256(6, 13),
            // ---------
            // Nested types
            // ---------
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Timestamp(TimeUnit::Nanosecond, None)),
            ),
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::FixedSizeBinary(23)),
            ),
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(
                    // nested dictionaries are probably a bad idea but they are possible
                    DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                ),
            ),
            DataType::Struct(Fields::from(vec![
                Field::new("f1", DataType::Int64, true),
                Field::new("f2", DataType::Float64, true),
                Field::new(
                    "f3",
                    DataType::Timestamp(TimeUnit::Second, Some("+08:00".into())),
                    true,
                ),
                Field::new(
                    "f4",
                    DataType::Dictionary(
                        Box::new(DataType::Int8),
                        Box::new(DataType::FixedSizeBinary(23)),
                    ),
                    true,
                ),
            ])),
            DataType::Struct(Fields::from(vec![
                Field::new("Int64", DataType::Int64, true),
                Field::new("Float64", DataType::Float64, true),
            ])),
            DataType::Struct(Fields::from(vec![
                Field::new("f1", DataType::Int64, true),
                Field::new(
                    "nested_struct",
                    DataType::Struct(Fields::from(vec![Field::new("n1", DataType::Int64, true)])),
                    true,
                ),
            ])),
            DataType::Struct(Fields::empty()),
            // TODO support more structured types (List, LargeList, Union, Map, RunEndEncoded, etc)
        ]
    }

    #[test]
    fn test_parse_data_type_whitespace_tolerance() {
        // (string to parse, expected DataType)
        let cases = [
            ("Int8", DataType::Int8),
            (
                "Timestamp        (Nanosecond,      None)",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "Timestamp        (Nanosecond,      None)  ",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "          Timestamp        (Nanosecond,      None               )",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "Timestamp        (Nanosecond,      None               )  ",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
        ];

        for (data_type_string, expected_data_type) in cases {
            println!("Parsing '{data_type_string}', expecting '{expected_data_type:?}'");
            let parsed_data_type = parse_data_type(data_type_string).unwrap();
            assert_eq!(parsed_data_type, expected_data_type);
        }
    }

    #[test]
    fn parse_data_type_errors() {
        // (string to parse, expected error message)
        let cases = [
            ("", "Unsupported type ''"),
            ("", "Error finding next token"),
            ("null", "Unsupported type 'null'"),
            ("Nu", "Unsupported type 'Nu'"),
            (
                r#"Timestamp(Nanosecond, Some(+00:00))"#,
                "Error unrecognized word: +00:00",
            ),
            (
                r#"Timestamp(Nanosecond, Some("+00:00))"#,
                r#"parsing "+00:00 as double quoted string: last char must be ""#,
            ),
            (
                r#"Timestamp(Nanosecond, Some(""))"#,
                r#"parsing "" as double quoted string: empty string isn't supported"#,
            ),
            (
                r#"Timestamp(Nanosecond, Some("+00:00""))"#,
                r#"parsing "+00:00"" as double quoted string: escaped double quote isn't supported"#,
            ),
            ("Timestamp(Nanosecond, ", "Error finding next token"),
            (
                "Float32 Float32",
                "trailing content after parsing 'Float32'",
            ),
            ("Int32, ", "trailing content after parsing 'Int32'"),
            ("Int32(3), ", "trailing content after parsing 'Int32'"),
            ("FixedSizeBinary(Int32), ", "Error finding i64 for FixedSizeBinary, got 'Int32'"),
            ("FixedSizeBinary(3.0), ", "Error parsing 3.0 as integer: invalid digit found in string"),
            // too large for i32
            ("FixedSizeBinary(4000000000), ", "Error converting 4000000000 into i32 for FixedSizeBinary: out of range integral type conversion attempted"),
            // can't have negative precision
            ("Decimal32(-3, 5)", "Error converting -3 into u8 for Decimal32: out of range integral type conversion attempted"),
            ("Decimal64(-3, 5)", "Error converting -3 into u8 for Decimal64: out of range integral type conversion attempted"),
            ("Decimal128(-3, 5)", "Error converting -3 into u8 for Decimal128: out of range integral type conversion attempted"),
            ("Decimal256(-3, 5)", "Error converting -3 into u8 for Decimal256: out of range integral type conversion attempted"),
            ("Decimal32(3, 500)", "Error converting 500 into i8 for Decimal32: out of range integral type conversion attempted"),
            ("Decimal64(3, 500)", "Error converting 500 into i8 for Decimal64: out of range integral type conversion attempted"),
            ("Decimal128(3, 500)", "Error converting 500 into i8 for Decimal128: out of range integral type conversion attempted"),
            ("Decimal256(3, 500)", "Error converting 500 into i8 for Decimal256: out of range integral type conversion attempted"),
            ("Struct(f1, Int64)", "Error finding next type, got unexpected ','"),
            ("Struct(f1 Int64,)", "Expected a word for the name of Struct, but got trailing comma"),
            ("Struct(f1)", "Error finding next type, got unexpected ')'"),
        ];

        for (data_type_string, expected_message) in cases {
            println!("Parsing '{data_type_string}', expecting '{expected_message}'");
            match parse_data_type(data_type_string) {
                Ok(d) => panic!("Expected error while parsing '{data_type_string}', but got '{d}'"),
                Err(e) => {
                    let message = e.to_string();
                    assert!(
                        message.contains(expected_message),
                        "\n\ndid not find expected in actual.\n\nexpected: {expected_message}\nactual:{message}\n"
                    );
                    // errors should also contain  a help message
                    assert!(message.contains("Must be a supported arrow type name such as 'Int32' or 'Timestamp(Nanosecond, None)'"));
                }
            }
        }
    }

    #[test]
    fn parse_error_type() {
        let err = parse_data_type("foobar").unwrap_err();
        assert!(matches!(err, ArrowError::ParseError(_)));
        assert_eq!(err.to_string(), "Parser error: Unsupported type 'foobar'. Must be a supported arrow type name such as 'Int32' or 'Timestamp(Nanosecond, None)'. Error unrecognized word: foobar");
    }
}
