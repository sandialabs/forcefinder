# SourcePathReceiver Types
SPR objects have been created in ForceFinder for three main response types:

```{glossary}
`LinearSourcePathReceiver`
    This SPR object type is intended for responses that are formatted as linear spectra, which are supplied as a SDynPy `SpectrumArray`

`PowerSourcePathReceiver`
    This SPR object type is intended for responses that are formatted as cross power spectral densities (CPSDs), which are supplied as a SDynPy `PowerSpectralDensityArray`

`TransientSourcePathReceiver`
    This SPR object type is intended for responses that are formatted as time traces, which are supplied as a SDynPy `TimeHistoryArray`
```

The "look and feel" for the programming interface is intended to be very similar for the different SPR object types. For example, the names for different inverse methods and class attributes are the same for the different SPR types. However, each SPR object type has unique rules that are applied when it is created (to validate common abscissa, etc.). Similarly, different methods, attributes, or inverse method options might exist for the different SPR types. Lastly, the form of the inverse problem is different for each SPR type. These differences will be explained as necessary throughout the documentation. 