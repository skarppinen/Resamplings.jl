const _INTENT_CHOICES = (:unconditional, :conditional);
function _check_intent(intent::Symbol)
    if !(intent in _INTENT_CHOICES)
        msg = "invalid intent `$intent`. Valid choices are $_INTENT_CHOICES.";
        throw(ArgumentError(msg));
    end
end

function _has_conditional_or_error(resampling::Symbol, randomisation::Symbol,
                                  order::Symbol, msg::AbstractString = "")
    if msg == ""
        msg = string(resampling, " resampling does not implement conditional resampling with ",
                     "randomisation = :", randomisation, " and order = :", order, ".");
    end
    if !has_conditional(resampling, randomisation, order)
        throw(ArgumentError(msg));
    end
end

for s in ("stratified", "systematic")
    eval(:(
        function Resampling{Symbol($s)}(N::Integer;
                                        randomisation::Symbol = :default,
                                        order::Symbol = :default,
                                        intent::Symbol = :unconditional,
                                        FT::DataType = Float64,
                                        IT::DataType = Int)
            _check_intent(intent);
            defaults = default_args(Resampling{Symbol($s)}, intent);
            settings = collect(pairs((randomisation = randomisation,
                                      order = order)));
            args = merge(defaults, NamedTuple(filter(p -> p.second != :default, settings)));
            if intent == :conditional
                _has_conditional_or_error(Symbol($s), args.randomisation, args.order);
            end
            Resampling{Symbol($s)}(N, args.randomisation, args.order, FT, IT, nothing);
        end
    ))
end

## Convenience constructor for multinomial resampling.
function MultinomialResampling(N::Integer;
                               randomisation::Symbol = :default,
                               intent::Symbol = :unconditional,
                               FT::DataType = Float64,
                               IT::DataType = Int)
    _check_intent(intent);
    defaults = default_args(MultinomialResampling, intent);
    settings = collect(pairs((randomisation = randomisation,)));
    args = merge(defaults, NamedTuple(filter(p -> p.second != :default, settings)));
    if intent == :conditional
        msg = string("multinomial resampling does not implement conditional resampling ",
                     "with randomisation = :$randomisation.");
        _has_conditional_or_error(:multinomial, args.randomisation, args.order, msg);
    end
    MultinomialResampling(N, args.randomisation, args.order, FT, IT, nothing);
end

## Convenience constructor for killing resampling.
function KillingResampling(N::Integer;
                           randomisation::Symbol = :default,
                           max_div = :default,
                           intent::Symbol = :unconditional,
                           FT::DataType = Float64,
                           IT::DataType = Int)
    _check_intent(intent);
    defaults = default_args(KillingResampling, intent);
    settings = collect(pairs((randomisation = randomisation,
                              max_div = max_div)));
    args = merge(defaults, NamedTuple(filter(p -> p.second != :default, settings)));
    if intent == :conditional
        # NOTE: order = :none always with killing resampling.
        msg = string("killing resampling does not implement conditional ",
                     "resampling with randomisation = :$randomisation.");
        _has_conditional_or_error(:killing, args.randomisation, args.order, msg);
    end
    KillingResampling(N, args.randomisation, args.order, FT, IT, (max_div = args.max_div,));
end

## Convenience constructors for residual, ssp and sss.
for s in ("residual", "sss", "ssp")
    eval(
        :(
            function Resampling{Symbol($s)}(N::Integer;
                                            randomisation::Symbol = :default,
                                            order::Symbol = :default,
                                            intent::Symbol = :unconditional,
                                            FT::DataType = Float64,
                                            IT::DataType = Int)
                _check_intent(intent);
                defaults = default_args(Resampling{Symbol($s)}, intent);
                settings = collect(pairs((randomisation = randomisation,
                                          order = order)));
                args = merge(defaults, NamedTuple(filter(p -> p.second != :default, settings)));
                if intent == :conditional
                    if Symbol($s) in (:residual, :ssp)
                        msg = string(Symbol($s), " resampling does not implement conditional resampling.");
                        throw(ArgumentError(msg));
                    end
                    _has_conditional_or_error(Symbol($s), args.randomisation, args.order);
                end
                Resampling{Symbol($s)}(N, args.randomisation, args.order, FT, IT, (w = zeros(FT, N),));
            end
        )
    )
end
